from keras.utils import Sequence
import os
import signal
import psutil
import pandas as pd
import numpy as np
import random
import threading 
from random import shuffle 
import math
from math import ceil, floor
import pysam
from ..util import *
from ..tiledb_config import * 
import tiledb
import pdb
from ..s3_sync import * 
from collections import OrderedDict
import gc 
import pdb             


def get_upsampled_indices_chrom(inputs):
    region_start=inputs[0]
    region_end=inputs[1]
    tdb_array_name=inputs[2]
    tdb_ambig_attribute=inputs[3]
    tdb_partition_attribute_for_upsample=inputs[4]
    task_indices=inputs[5]
    tdb_partition_thresh_for_upsample=inputs[6]
    print("starting getting indices to upsample in range:"+str(region_start)+"-"+str(region_end))
    with tiledb.open(tdb_array_name,'r',ctx=tiledb.Ctx(get_default_config())) as tdb_array:
        if tdb_ambig_attribute is not None:
            attr_vals=tdb_array.query(attrs=[tdb_ambig_attribute,tdb_partition_attribute_for_upsample]).multi_index[region_start:region_end-1,task_indices]
            ambig_attr_vals=np.sum(attr_vals[tdb_ambig_attribute],axis=1)
        else:
            attr_vals=tdb_array.query(attrs=[tdb_partition_attribute_for_upsample]).multi_index[region_start:region_end-1,task_indices]        
        upsample_vals=np.sum(attr_vals[tdb_partition_attribute_for_upsample],axis=1)
    if tdb_ambig_attribute is not None:
        cur_upsampled_indices=region_start+np.argwhere((upsample_vals>=tdb_partition_thresh_for_upsample) & ( ambig_attr_vals==0))
    else: 
        cur_upsampled_indices=region_start+np.argwhere(upsample_vals>=tdb_partition_thresh_for_upsample)
    print("finished indices to upsample in range:"+str(region_start)+"-"+str(region_end))
    return cur_upsampled_indices

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)



class TiledbGenerator(Sequence):
    def __init__(self,
                 ref_fasta,
                 batch_size,
                 tdb_array,
                 tdb_partition_attribute_for_upsample,
                 tdb_partition_thresh_for_upsample,
                 tdb_input_source_attribute,
                 tdb_input_flank,
                 upsample_ratio,
                 tdb_output_source_attribute,
                 tdb_output_flank,
                 num_inputs,
                 num_outputs,
                 tdb_input_min=None,
                 tdb_input_max=None,
                 tdb_output_min=None,
                 tdb_output_max=None,
                 task_indices=None,
                 tasks=None,
                 tdb_input_aggregation=None,
                 tdb_input_transformation=None,
                 tdb_output_aggregation=None,
                 tdb_output_transformation=None,
                 tdb_ambig_attribute=None,
                 tdb_bias_arrays=None,
                 tdb_bias_source_attribute=None,
                 tdb_bias_flank=None,
                 tdb_bias_aggregation=None,
                 tdb_bias_transformation=None,
                 chroms=None,
                 chrom_sizes=None,
                 shuffle_epoch_start=True,
                 shuffle_epoch_end=True,
                 pseudocount=0,
                 add_revcomp=False,
                 expand_dims=False,
                 return_coords=False,
                 tdb_config=None,
                 tdb_ctx=None,
                 num_threads=1):
        '''
        tdb_partition_attribute_for_upsample -- attribute in tiledb array used for determining which bases to upsample (usu. 'idr_peak') 
        tdb_partition_thresh_for_upsample -- threshold for determinining samples to upsample (generally 1) 
        tdb_input_aggregation/ tdb_output_aggregation -- one of 'average','max','binary_max','sum',None
        '''
        self.num_threads=num_threads
        self.shuffle_epoch_start=shuffle_epoch_start
        self.shuffle_epoch_end=shuffle_epoch_end

        #get local copy of s3 reference sequence
        if ref_fasta.startswith('s3://'):
            self.ref_fasta=download_s3_file(ref_fasta)
            fai=download_s3_file(ref_fasta+'.fai')
        else: 
            self.ref_fasta=ref_fasta

        self.batch_size=batch_size

        self.add_revcomp=add_revcomp
        if self.add_revcomp==True:
            self.batch_size=int(math.floor(self.batch_size/2))
            
        self.expand_dims=expand_dims

        #create tiledb configuration parameters (these have been found optimal for most use cases, but should set in a separate config file in the future)
        if tdb_config is not None:
            self.config=tdb_config
        else:
            self.config=get_default_config()
        if tdb_ctx is not None:
            self.ctx=tdb_ctx
        else:
            self.ctx=tiledb.Ctx(self.config)
            
        print("opening:"+tdb_array+" for reading...")
        self.tdb_array_name=tdb_array
        self.tdb_array=tiledb.open(tdb_array,mode='r',ctx=self.ctx)
        if tdb_bias_arrays is not None:
            self.bias_arrays=[tiledb.open(tdb_bias_arrays[i],mode='r',ctx=self.ctx) for i in range(len(tdb_bias_arrays))]
            self.bias_source_attribute=tdb_bias_source_attribute
            self.bias_flank=tdb_bias_flank
            self.bias_aggregation=tdb_bias_aggregation
            self.bias_transformation=tdb_bias_transformation 
        else:
            self.bias_arrays=None
        print("success!")

        #identify chromosome information
        if chroms is not None:
            self.chroms_to_use=chroms
        else:
            if chrom_sizes.startswith("s3://"):
                self.chroms_to_use=[i.split()[0] for i in read_s3_file_contents(chrom_sizes).strip().split('\n')]
            else:
                self.chroms_to_use=[i.split()[0] for i in open(chrom_sizes,'r').read().strip().split('\n')]
        #find the tdb indices that correspond to the chroms to be used 
        self.get_chrom_index_ranges(self.chroms_to_use)
        print("self.weighted_chrom_indices"+str(self.weighted_chrom_indices))
        print("got indices for used chroms")

        #get indices of tasks to be used in training
        if tasks is not None:
            self.task_indices=self.get_task_indices(tasks)
        elif task_indices is not None:
            self.task_indices=task_indices
        else:
            #already got task indices when calling get_chrom_index_ranges function above
            pass 
        print("identified task indices:"+str(self.task_indices))
        self.tdb_ambig_attribute=tdb_ambig_attribute

        #store input params
        self.num_inputs=num_inputs
        self.tdb_input_source_attribute=tdb_input_source_attribute
        self.tdb_input_flank=tdb_input_flank
        self.tdb_input_aggregation=[str(i) for i in tdb_input_aggregation]
        self.tdb_input_transformation=[str(i) for i in tdb_input_transformation]

        #store output params
        self.num_outputs=num_outputs
        self.tdb_output_source_attribute=tdb_output_source_attribute
        self.tdb_output_flank=tdb_output_flank
        self.tdb_output_aggregation=[str(i) for i in tdb_output_aggregation]
        self.tdb_output_transformation=[str(i) for i in tdb_output_transformation]


        #identify min/max values
        self.tdb_input_min=transform_data_type(tdb_input_min,self.num_inputs)
        self.tdb_input_max=transform_data_type(tdb_input_max,self.num_inputs)
        self.tdb_output_min=transform_data_type(tdb_output_min,self.num_outputs)
        self.tdb_output_max=transform_data_type(tdb_output_max,self.num_outputs)
                
        #identify upsampled genome indices for model training
        self.tdb_partition_attribute_for_upsample=tdb_partition_attribute_for_upsample
        self.tdb_partition_thresh_for_upsample=tdb_partition_thresh_for_upsample
        if upsample_ratio is not None:
            assert type(upsample_ratio)==float
        self.upsample_ratio=upsample_ratio
        if self.upsample_ratio is not None:
            self.get_upsampled_indices()
            self.upsampled_batch_size=math.ceil(self.upsample_ratio*self.batch_size)
        else:
            self.upsampled_batch_size=0
            self.upsampled_indices_len=0
            self.upsampled_indices=[]
            

        self.non_upsampled_batch_size=self.batch_size-self.upsampled_batch_size        

        self.pseudocount=pseudocount
        self.return_coords=return_coords
        print('created generator')
        
    def get_chrom_index_ranges(self,chroms_to_use):
        '''
        find tdb indices corresponding to the used chromosomes 
        '''
        num_chroms=self.tdb_array.meta['num_chroms']
        self.task_indices=[i for i in range(self.tdb_array.meta['num_tasks'])]
        chrom_indices=[]
        chrom_sizes=[]
        chroms=[]
        num_indices=0
        for i in range(num_chroms):
            chrom_name=self.tdb_array.meta['chrom_'+str(i)]
            if chrom_name in chroms_to_use:
                chroms.append(chrom_name)
                start_index=self.tdb_array.meta['offset_'+str(i)]
                end_index=start_index+self.tdb_array.meta['size_'+str(i)]
                num_indices+=(end_index-start_index)
                chrom_indices.append((start_index,end_index))
                chrom_sizes.append(self.tdb_array.meta['size_'+str(i)])
        min_chrom_size=min(chrom_sizes)
        scaled_chrom_sizes=[round(i/min_chrom_size) for i in chrom_sizes]
        weighted_chrom_sizes=[]
        for i in range(len(chrom_sizes)):
            cur_weight=scaled_chrom_sizes[i]
            cur_range=[chrom_indices[i]]
            weighted_chrom_sizes=weighted_chrom_sizes+cur_weight*cur_range
        self.chrom_indices=chrom_indices
        self.weighted_chrom_indices=weighted_chrom_sizes
        self.num_indices=num_indices
        self.chroms_to_use=chroms
        return
    
    def get_task_indices(self,tasks):
        '''
        get tdb indices of user-specified tasks 
        '''
        num_tasks=self.tdb_array.meta['num_tasks']
        task_indices=[]
        for i in range(num_tasks):
            cur_task=self.tdb_array.meta['task_'+str(i)]
            if cur_task in tasks:
                task_indices.append(i)
        assert(len(task_indices)>0)
        return task_indices
    
    def get_nonupsample_batch_indices(self):
        '''
        randomly select n positions from the genome 
        '''
        #get current chromosome
        cur_interval=random.sample(self.weighted_chrom_indices,1)[0]
        #sample random indices from the current chromosome 
        cur_batch=random.sample(range(cur_interval[0],cur_interval[1]),self.non_upsampled_batch_size)
        return cur_batch
                                                                                                    
    

    def get_upsampled_indices(self):
        from multiprocessing import Pool
        print("num_threads:"+str(self.num_threads))
        pool=Pool(processes=self.num_threads,initializer=init_worker)
        pool_inputs=[] 
        for region in self.chrom_indices:
            region_start=region[0]
            region_end=region[1]
            pool_inputs.append((region_start,region_end,self.tdb_array_name,self.tdb_ambig_attribute,self.tdb_partition_attribute_for_upsample,self.task_indices,self.tdb_partition_thresh_for_upsample))
        upsampled_indices=None
        try:
            for region_upsampled_indices in pool.map(get_upsampled_indices_chrom,pool_inputs):
                if upsampled_indices is None:
                    upsampled_indices=np.squeeze(region_upsampled_indices)
                else:
                    upsampled_indices=np.concatenate((upsampled_indices,np.squeeze(region_upsampled_indices)))
        except KeyboardInterrupt:
            kill_child_processes(os.getpid())
            pool.terminate()
            raise
        except Exception as e:
            print(e)
            kill_child_processes(os.getpid())
            raise 
        pool.close()
        pool.join()
        print('closed upsampling pool') 
        print("made upsampled index data frame")
        self.upsampled_indices=upsampled_indices
        if self.shuffle_epoch_start==True:
            #shuffle rows & reset index
            print("shuffling upsampled dataframes prior to start of training")
            np.random.shuffle(self.upsampled_indices)
        self.upsampled_indices_len=len(self.upsampled_indices)
        print("finished upsampling")
        return

    
        
    def __len__(self):
        #we are only training on peak regions
        if (self.upsample_ratio is not None) and (self.upsample_ratio==1):
            return int(floor(self.upsampled_indices_len/self.upsampled_batch_size))
        else:
        #training on peak and non-peak regions 
            return int(floor(self.num_indices/self.batch_size))
    

    def __getitem__(self,idx):
        gc.unfreeze()
        self.ref=pysam.FastaFile(self.ref_fasta)
        
        #get the coordinates for the current batch
        tdb_batch_indices=self.get_tdb_indices_for_batch(idx) #coords is a df with 'chrom' and 'pos' columns.
        coords=None
        if self.return_coords is True:
            #get the chromosome coordinates that correspond to indices
            coords=self.get_coords(tdb_batch_indices)
        #get the inputs 
        X=[]
        for cur_input_index in range(self.num_inputs):
            cur_input=self.tdb_input_source_attribute[cur_input_index]
            if cur_input=="seq":                
                #get the one-hot encoded sequence
                if coords is None:
                    coords=self.get_coords(tdb_batch_indices)
                cur_seq=self.get_seq(coords,self.tdb_input_flank[cur_input_index])
                transformed_seq=self.transform_seq(cur_seq,self.tdb_input_transformation[cur_input_index])
                cur_x=one_hot_encode(transformed_seq)
            else:
                #extract values from tdb
                cur_vals=self.get_tdb_vals(tdb_batch_indices,cur_input_index,self.tdb_input_flank[cur_input_index],is_input=True)
                aggregate_vals=self.aggregate_vals(cur_vals,self.input_aggregation[cur_input_index])
                transformed_vals=self.transform_vals(aggregate_vals,self.tdb_input_transformation[cur_input_index])
                cur_x=transformed_vals
            if self.expand_dims==True:
                cur_x=np.expand_dims(cur_x,axis=1)
            X.append(cur_x)

        #get the biases, if specified
        if self.bias_arrays is not None:
            for cur_bias_index in range(len(self.bias_arrays)):
                cur_bias_vals=self.get_bias_vals(tdb_batch_indices,cur_bias_index,self.bias_flank[cur_bias_index])
                aggregate_bias=self.aggregate_vals(cur_bias_vals,self.bias_aggregation[cur_bias_index])
                transformed_bias=self.transform_vals(aggregate_bias,self.bias_transformation[cur_bias_index])
                cur_bias=transformed_bias
                if self.expand_dims==True:
                    cur_bias=np.expand_dims(cur_bias,axis=1)
                X.append(cur_bias) 
            
        #get the outputs 
        y=[]
        for cur_output_index in range(self.num_outputs):
            cur_output=self.tdb_output_source_attribute[cur_output_index]
            if cur_output=="seq":
                #get the one-hot encoded sequence
                if coords is None:
                    coords=get_coords(tdb_batch_indices)
                cur_seq=self.get_seq(coords,self.tdb_output_flank[cur_output_index])
                transformed_seq=self.transform_seq(cur_seq,self.tdb_output_transformation[cur_output_index])
                cur_y=one_hot_encode(transformed_seq)
            else:
                #extract values from tdb
                cur_vals=self.get_tdb_vals(tdb_batch_indices,cur_output_index,self.tdb_output_flank[cur_output_index],is_output=True)
                aggregate_vals=self.aggregate_vals(cur_vals,self.tdb_output_aggregation[cur_output_index])
                transformed_vals=self.transform_vals(aggregate_vals,self.tdb_output_transformation[cur_output_index])
                cur_y=transformed_vals
            y.append(cur_y)
        if self.return_coords is True:
            if self.add_revcomp==True:
                coords=coords+coords #concatenate coord list 
        
        filtered_X,filtered_y,filtered_coords=self.remove_data_out_of_range(X,y,coords)
        
        if self.return_coords is True:
            return (filtered_X,filtered_y,filtered_coords)
        else:
            return (filtered_X,filtered_y)
        
    def remove_data_out_of_range(self,X,y,coords=None):
        bad_indices=[]
        for i in range(self.num_inputs):
            if self.tdb_input_min[i] is not None:
                out_of_range=[i[0] for i in np.argwhere(X[:,i]<self.tdb_input_min[i]).tolist() if len(i)>0 ]
                bad_indices+=out_of_range
            if self.tdb_input_max[i] is not None:
                out_of_range=[i[0] for i in np.argwhere(X[:,i]>self.tdb_input_max[i]).tolist() if len(i)>0 ]
                bad_indices+=out_of_range
        for i in range(self.num_outputs):
            if self.tdb_output_min[i] is not None:
                out_of_range=[i[0] for i in np.argwhere(y[:,i]<self.tdb_output_min[i]).tolist() if len(i)>0]
                bad_indices+=out_of_range
            if self.tdb_output_max[i] is not None:
                out_of_range=[i[0] for i in np.argwhere(y[:,i]<self.tdb_output_max[i]).tolist() if len(i)>0]
                bad_indices+=out_of_range
        X=[np.delete(i,bad_indices,0) for i in X]
        y=[np.delete(i,bad_indices,0) for i in y]
        if coords is not None:
            coords=np.delete(coords,bad_indices,0)
        return X,y,coords
        
    def get_coords(self,tdb_batch_indices):
        #return list of (chrom,pos) for each index in batch
        coords=[]
        for cur_batch_index in tdb_batch_indices:
            for chrom_index in range(len(self.chrom_indices)):
                cur_chrom_start_index=self.chrom_indices[chrom_index][0]
                cur_chrom_end_index=self.chrom_indices[chrom_index][1]
                if (cur_batch_index >=cur_chrom_start_index) and (cur_batch_index<cur_chrom_end_index):
                    coords.append([self.chroms_to_use[chrom_index],cur_batch_index-cur_chrom_start_index])
                    break
        return coords
    
    def get_tdb_indices_for_batch(self,idx):
        upsampled_batch_indices=None
        non_upsampled_batch_indices=None
        if self.upsampled_batch_size > 0:
            #might need to wrap to get the upsampled index length
            upsampled_batch_start=int(idx*self.upsampled_batch_size % self.upsampled_indices_len)
            upsampled_batch_end=upsampled_batch_start+self.upsampled_batch_size
            while upsampled_batch_end > self.upsampled_indices_len:
                if upsampled_batch_indices is None:
                    upsampled_batch_indices=self.upsampled_indices[upsampled_batch_start:self.upsampled_indices_len]
                else:
                    upsampled_batch_indices=np.concatenate((upsampled_batch_indices,self.upsampled_indices[upsampled_batch_start:self.upsampled_indices_len]))
                upsampled_batch_start=0
                upsampled_batch_end=upsampled_batch_end-self.upsampled_indices_len
            if upsampled_batch_indices is None:
                upsampled_batch_indices=self.upsampled_indices[upsampled_batch_start:upsampled_batch_end]
            else: 
                upsampled_batch_indices=np.concatenate((upsampled_batch_indices,self.upsampled_indices[upsampled_batch_start:upsampled_batch_end]))
            
        if self.non_upsampled_batch_size > 0:
            #select random indices from genome
            non_upsampled_batch_indices=self.get_nonupsample_batch_indices()
        if (upsampled_batch_indices is not None) and (non_upsampled_batch_indices is not None):
            tdb_batch_indices=np.concatenate((upsampled_batch_indices,non_upsampled_batch_indices))
        elif upsampled_batch_indices is not None:
            tdb_batch_indices=upsampled_batch_indices
        elif non_upsampled_batch_indices is not None:
            tdb_batch_indices=non_upsampled_batch_indices
        else:
            raise Exception("both upsampled_batch_indices and non_upsampled_batch_indices appear to be none")
        return tdb_batch_indices
    
     
    def get_seq(self,coords,flank):
        seqs=[]
        for coord in coords:
            chrom=coord[0]
            start_pos=coord[1]-flank
            end_pos=coord[1]+flank 
            try:
                seq=self.ref.fetch(chrom,start_pos,end_pos)
                if len(seq)<2*flank:
                    delta=2*flank-len(seq)
                    seq=seq+"N"*delta
            except:
                seq="N"*2*flank
            seqs.append(seq) 
        return seqs

    def transform_seq(self,seqs,transformation):
        if self.add_revcomp is True:
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        return seqs

    def get_bias_vals(self,tdb_batch_indices,cur_bias_index,flank):
        num_entries=len(tdb_batch_indices)
        vals=np.full((num_entries,2*flank,1),np.nan)
        cur_array=self.bias_arrays[cur_bias_index]
        for val_index in range(num_entries):
            vals[val_index,:,:]=cur_array.query(attrs=[self.bias_source_attribute[cur_bias_index]])[tdb_batch_indices[val_index]-flank:tdb_batch_indices[val_index]+flank,:][self.bias_source_attribute[cur_bias_index]]
        return vals 
    
    def get_tdb_vals(self,tdb_batch_indices,input_output_index,flank,is_input=False,is_output=False):
        '''
        extract the values from tileDB 
        '''
        #determine the attribute from tiledb that will be used 
        assert is_input==True or is_output==True
        if is_input==True:
            input_or_output="inputs"
            attribute=self.tdb_input_source_attribute[input_output_index]
        else:
            input_or_output="outputs"
            attribute=self.tdb_output_source_attribute[input_output_index] 
            
        num_tasks=len(self.task_indices)
        num_entries=len(tdb_batch_indices)
        #prepopulate the values array with nans
        vals=np.full((num_entries,2*flank,num_tasks),np.nan)
        #iterate through entries
        for val_index in range(num_entries):
            vals[val_index,:,:]=self.tdb_array.query(attrs=[attribute]).multi_index[tdb_batch_indices[val_index]-flank:tdb_batch_indices[val_index]+flank-1,self.task_indices][attribute]
        return vals
    
    def transform_vals(self,vals,transformer):
        if self.add_revcomp==True:
            vals=np.concatenate((vals,vals),axis=0)
        if transformer == 'None':
            return vals
        elif transformer == 'asinh':
            return np.arcsinh(vals)
        elif transformer == 'log10':
            return np.log10(vals+self.pseudocount)
        elif transformer == 'log':
            return np.log(vals+self.pseudocount)
        else:
            raise Exception("transform_vals argument must be one of None, asinh, log10, log; you provided:"+transformer) 
    
    def aggregate_vals(self,vals,aggregator):
        if aggregator == 'None':
            return vals
        elif aggregator == 'average':
            return np.mean(vals,axis=1)
        elif aggregator == 'max':
            return np.max(vals,axis=1)
        elif aggregator == 'binary_max':
            #get the max in the interval, but cap it at one or 0
            raw_max=np.max(vals,axis=1)
            raw_max[raw_max>1]=1
            raw_max[raw_max<0]=0
            return raw_max
        elif aggregator == 'sum':
            return np.sum(vals,axis=1) 
        else:
            raise Exception("aggregate_vals argument must be one of None, average, max, sum; you provided:"+aggregator)

    
    def on_epoch_end(self):
        if self.shuffle_epoch_end==True:
            print("WARNING: SHUFFLING ON EPOCH END MAYBE SLOW:"+str(self.upsampled_indices.shape))
            self.upsampled_indices=self.upsampled_indices.sample(frac=1)

