from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
import threading 
from random import shuffle 
import math
from math import ceil, floor
import pysam
from ..util import *
import tiledb
import pdb
from collections import OrderedDict

def get_genome_size(chrom_sizes_file,chroms):
    '''
    get size of chromosomes to train on 
    '''
    chrom_sizes=pd.read_csv(chrom_sizes_file,header=None,sep='\t')
    chrom_sizes_subset=chrom_sizes[chrom_sizes[0].isin(chroms)]
    chrom_sizes_subset_dict=dict() 
    genome_size=chrom_sizes_subset[1].sum()
    last_index_to_chrom=dict()
    last_index=0
    for index,row in chrom_sizes_subset.iterrows():
        chrom_name=row[0]
        chrom_size=row[1]
        chrom_sizes_subset_dict[chrom_name]=chrom_size
        first_index=last_index 
        last_index+=chrom_size
        last_index_to_chrom[last_index]=[chrom_name,chrom_size,first_index]
    return chrom_sizes_subset_dict,last_index_to_chrom, genome_size

            

class TiledbGenerator(Sequence):
    def __init__(self,
                 ref_fasta,
                 batch_size,
                 tdb_indexer,
                 tdb_partition_attribute_for_upsample,
                 tdb_partition_thresh_for_upsample,
                 tdb_inputs,
                 tdb_input_source_attribute,
                 tdb_input_flank,
                 upsample_ratio,
                 tdb_outputs,
                 tdb_output_source_attribute,
                 tdb_output_flank,
                 num_inputs,
                 num_outputs,
                 tdb_input_aggregation=None,
                 tdb_input_transformation=None,
                 tdb_output_aggregation=None,
                 tdb_output_transformation=None,
                 chroms=None,
                 chrom_sizes=None,
                 shuffle_epoch_start=True,
                 shuffle_epoch_end=True,
                 pseudocount=0,
                 add_revcomp=False,
                 expand_dims=False,
                 return_coords=False):
        '''
        tdb_partition_attribute_for_upsample -- attribute in tiledb array used for determining which bases to upsample (usu. 'idr_peak') 
        tdb_partition_thresh_for_upsample -- threshold for determinining samples to upsample (generally 1) 
        tdb_input_aggregation/ tdb_output_aggregation -- one of 'avg','max',None
        '''
        self.shuffle_epoch_start=shuffle_epoch_start
        self.shuffle_epoch_end=shuffle_epoch_end
        self.ref_fasta=ref_fasta
        self.batch_size=batch_size
        self.add_revcomp=add_revcomp
        if self.add_revcomp==True:
            self.batch_size=int(math.floor(self.batch_size/2))
            
        self.expand_dims=expand_dims
        
        #identify chromosome information
        if chroms is not None:
            self.chroms_to_use=chroms
        else: 
            self.chroms_to_use=[i.split()[0] for i in open(chrom_sizes,'r').read().strip().split('\n')]            
        self.chrom_sizes,self.last_index_to_chrom,self.length=get_genome_size(chrom_sizes,self.chroms_to_use)
        
        #identify indices to upsample
        self.tdb_indexer=tdb_indexer

        #store input params
        self.num_inputs=num_inputs
        self.tdb_inputs=tdb_inputs
        self.tdb_input_source_attribute=tdb_input_source_attribute
        self.tdb_input_flank=tdb_input_flank
        self.tdb_input_aggregation=[str(i) for i in tdb_input_aggregation]
        self.tdb_input_transformation=[str(i) for i in tdb_input_transformation]

        #store output params
        self.num_outputs=num_outputs
        self.tdb_outputs=tdb_outputs
        self.tdb_output_source_attribute=tdb_output_source_attribute
        self.tdb_output_flank=tdb_output_flank
        self.tdb_output_aggregation=[str(i) for i in tdb_output_aggregation]
        self.tdb_output_transformation=[str(i) for i in tdb_output_transformation]
        self.data_arrays=self.open_tiledb_arrays_for_reading()        


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

            
    def open_tiledb_arrays_for_reading(self):
        '''
        Opens tiledb arrays for each task/chromosome for reading  
        '''
        array_dict=OrderedDict()
        array_dict['index']=OrderedDict()
        array_dict['inputs']=OrderedDict()
        array_dict['outputs']=OrderedDict() 

        #index tasks 
        index_tasks=open(self.tdb_indexer).read().strip().split('\n')
        for task in index_tasks:
            array_dict['index'][task]=OrderedDict() 
            for chrom in self.chroms_to_use:
                array_dict['index'][task][chrom]=task+"."+chrom
        #inputs 
        for i in range(len(self.tdb_inputs)):
            tdb_input=self.tdb_inputs[i] 
            if tdb_input=="seq":
                continue 
            tdb_input_tasks=open(tdb_input).read().strip().split('\n')
            tdb_input_array=OrderedDict() 
            for task in tdb_input_tasks:
                tdb_input_array[task]=OrderedDict() 
                for chrom in self.chroms_to_use:
                    tdb_input_array[task][chrom]=task+'.'+chrom
            array_dict['inputs'][i]=tdb_input_array
            
        #outputs
        for i in range(len(self.tdb_outputs)): 
            tdb_output=self.tdb_outputs[i]
            if tdb_output=="seq":
                continue 
            tdb_output_tasks=open(tdb_output).read().strip().split('\n')
            tdb_output_array=OrderedDict()
            for task in tdb_output_tasks:
                tdb_output_array[task]=OrderedDict() 
                for chrom in self.chroms_to_use:
                    tdb_output_array[task][chrom]=task+'.'+chrom
            array_dict['outputs'][i]=tdb_output_array 
        return array_dict
    
    def get_nonupsample_batch_indices(self):
        '''
        randomly select n positions from the genome 
        '''
        indices=random.sample(range(self.length),self.non_upsampled_batch_size)
        #get the chroms and coords for each index
        chroms=[]
        chrom_pos=[]
        for cur_index in indices:
            for chrom_last_index in self.last_index_to_chrom:
                if cur_index < chrom_last_index:
                    #this is the chromosome to use!
                    #make sure we don't slide off the edge of the chromosome 
                    cur_chrom,cur_chrom_size,cur_chrom_first_index=self.last_index_to_chrom[chrom_last_index]
                    cur_chrom_pos=random.randint(self.chrom_edge_flank, cur_chrom_size-self.chrom_edge_flank)
                    chroms.append(cur_chrom)
                    chrom_pos.append(cur_chrom_pos)
                    break 
        cur_batch= pd.DataFrame({'chrom':chroms,'pos':chrom_pos})
        assert cur_batch.shape[0]==self.non_upsampled_batch_size
        return cur_batch
                                                                                                    
        
    def get_upsampled_indices(self):
        #use pandas dataframes to store index,chrom,position for upsampled and non-upsampled values
        upsampled_chroms=None
        upsampled_indices=None
        self.chrom_edge_flank=max([max(self.tdb_input_flank),max(self.tdb_output_flank)])
        for chrom in self.chroms_to_use:
            upsampled_indices_chrom=None
            chrom_size=None
            for task in self.data_arrays['index']:
                with tiledb.DenseArray(self.data_arrays['index'][task][chrom], mode='r') as cur_array:
                    cur_vals=cur_array[:][self.tdb_partition_attribute_for_upsample]
                if chrom_size is None:
                    chrom_size=cur_vals.shape[0]
                print("got values for cur task/chrom") 
                upsampled_indices_task_chrom=np.argwhere(cur_vals>=self.tdb_partition_thresh_for_upsample)
                print("got upsampled indices")
                if upsampled_indices_chrom is None:
                    upsampled_indices_chrom=upsampled_indices_task_chrom
                else:
                    upsampled_indices_chrom=np.union1d(upsampled_indices_chrom,upsampled_indices_task_chrom)
                print("performed task union")
                
            #make sure we dont' run off the edges of the chromosome!
            usampled_indices_chrom=upsampled_indices_chrom[upsampled_indices_chrom>self.chrom_edge_flank]
            upsampled_indices_chrom=upsampled_indices_chrom[upsampled_indices_chrom<(self.chrom_sizes[chrom]-self.chrom_edge_flank)]
            print("got indices to upsample for chrom:"+str(chrom))            
            if upsampled_chroms is None:
                upsampled_chroms=[chrom]*upsampled_indices_chrom.shape[0]
                upsampled_indices=upsampled_indices_chrom
            else:
                upsampled_chroms=upsampled_chroms+[chrom]*upsampled_indices_chrom.shape[0]
                upsampled_indices=np.concatenate((upsampled_indices,upsampled_indices_chrom),axis=0)
            print("appended chrom indices to master list") 

        upsampled_indices=pd.DataFrame.from_dict({'chrom':upsampled_chroms,
                                                  'pos':upsampled_indices.squeeze()})

        print("made upsampled index data frame")
        if self.shuffle_epoch_start==True:
            #shuffle rows & reset index
            upsampled_indices=upsampled_indices.sample(frac=1)
            upsampled_indices=upsampled_indices.reset_index(drop=True)
            print("shuffling upsampled dataframes prior to start of training")
        self.upsampled_indices=upsampled_indices    
        self.upsampled_indices_len=len(self.upsampled_indices)
        num_pos_wraps=math.ceil(self.length/self.upsampled_indices_len)
        self.upsampled_indices=pd.concat([self.upsampled_indices]*num_pos_wraps, ignore_index=True)[0:self.length]
        return 

        
    def __len__(self):
        return int(floor(self.length/self.batch_size))
    

    def __getitem__(self,idx):
        self.ref=pysam.FastaFile(self.ref_fasta)
        
        #get the coordinates for the current batch
        coords=self.get_coords(idx) #coords is a df with 'chrom' and 'pos' columns.

        #get the inputs 
        X=[]
        for cur_input_index in range(self.num_inputs):
            cur_input=self.tdb_inputs[cur_input_index]
            if cur_input=="seq":                
                #get the one-hot encoded sequence
                cur_seq=self.get_seq(coords,self.tdb_input_flank[cur_input_index])
                transformed_seq=self.transform_seq(cur_seq,self.tdb_input_transformation[cur_input_index])
                cur_x=one_hot_encode(transformed_seq)
            else:
                #extract values from tdb
                cur_vals=self.get_tdb_vals(coords,cur_input_index,self.tdb_input_flank[cur_input_index],is_input=True)
                transformed_vals=self.transform_vals(cur_vals,self.tdb_input_transformation[cur_input_index])
                aggregate_vals=self.aggregate_vals(transformed_vals,self.input_aggregation[cur_input_index])
                cur_x=aggregate_vals
                
            if self.expand_dims==True:
                cur_x=np.expand_dims(cur_x,axis=1)
            X.append(cur_x)
            
        #get the outputs 
        y=[]
        for cur_output_index in range(self.num_outputs):
            cur_output=self.tdb_outputs[cur_output_index]
            if cur_output=="seq":
                #get the one-hot encoded sequence
                cur_seq=self.get_seq(coords,self.tdb_output_flank[cur_output_index])
                transformed_seq=self.transform_seq(cur_seq,self.tdb_output_transformation[cur_output_index])
                cur_y=one_hot_encode(transformed_seq)
            else:
                #extract values from tdb
                cur_vals=self.get_tdb_vals(coords,cur_output_index,self.tdb_output_flank[cur_output_index],is_output=True)
                transformed_vals=self.transform_vals(cur_vals,self.tdb_output_transformation[cur_output_index])
                aggregate_vals=self.aggregate_vals(transformed_vals,self.tdb_output_aggregation[cur_output_index])
                cur_y=aggregate_vals
            y.append(cur_y)
            
        if self.return_coords is True:
            if self.add_revcomp==True:
                coords=pd.concat((coords,coords),axis=0)
            coords=[(row[0],row[1]) for index,row in coords.iterrows()]
            return (X,y,coords)
        else:
            return (X,y) 
    
    def get_coords(self,idx):
        upsampled_batch_start=idx*self.upsampled_batch_size
        upsampled_batch_end=upsampled_batch_start+self.upsampled_batch_size
        upsampled_batch_indices=None
        non_upsampled_batch_indices=None
        if self.upsampled_batch_size > 0:
            upsampled_batch_indices=self.upsampled_indices.loc[list(range(upsampled_batch_start,upsampled_batch_end))]
        if self.non_upsampled_batch_size > 0:
            #select random indices from genome
            non_upsampled_batch_indices=self.get_nonupsample_batch_indices()
        if (upsampled_batch_indices is not None) and (non_upsampled_batch_indices is not None):
            coords=pd.concat((upsampled_batch_indices,non_upsampled_batch_indices),axis=0)
        elif upsampled_batch_indices is not None:
            coords=upsampled_batch_indices
        elif non_upsampled_batch_indices is not None:
            coords=non_upsampled_batch_indices
        else:
            raise Exception("both upsampled_batch_indices and non_upsampled_batch_indices appear to be none")
        return coords
    
     
    def get_seq(self,coords,flank):
        chroms=coords['chrom']
        start_pos=coords['pos']-flank
        end_pos=coords['pos']+flank
        seqs=[]
        for i in range(coords.shape[0]):
            try:
                seq=self.ref.fetch(chroms.iloc[i],start_pos.iloc[i],end_pos.iloc[i])
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
    
    def get_tdb_vals(self,coords,input_output_index,flank,is_input=False,is_output=False):
        '''
        extract the values from tileDB 
        '''
        assert is_input==True or is_output==True
        if is_input==True:
            input_or_output="inputs"
            attribute=self.tdb_input_source_attribute[input_output_index]
        else:
            input_or_output="outputs"
            attribute=self.tdb_output_source_attribute[input_output_index] 
            
        chroms=coords['chrom']
        start_positions=coords['pos']-flank
        end_positions=coords['pos']+flank
        
        task_chrom_to_tdb=self.data_arrays[input_or_output][input_output_index]
        tasks=list(task_chrom_to_tdb.keys())
        num_tasks=len(tasks)
        num_entries=coords.shape[0]
        #prepopulate the values array with 0
        vals=np.full((num_entries,2*flank,num_tasks),np.nan)
        #define a context for querying tiledb
        ctx=tiledb.Ctx()
        #iterate through entries 
        for val_index in range(num_entries):            
            #iterate through tasks for each entry 
            for task_index in range(num_tasks):
                task=tasks[task_index] 
                chrom=chroms.iloc[val_index]
                start_position=start_positions.iloc[val_index]
                if np.isnan(start_position):
                    continue
                end_position=end_positions.iloc[val_index]
                if np.isnan(end_position):
                    continue 
                
                array_name=task_chrom_to_tdb[task][chrom]
                with tiledb.DenseArray(array_name, mode='r',ctx=ctx) as cur_array:
                    try:
                        cur_vals=cur_array[int(start_position):int(end_position)][attribute]
                        vals[val_index,:,task_index]=cur_vals
                    except:
                        print("skipping: start_position:"+str(start_position)+" : end_position:"+str(end_position))
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
        else:
            raise Exception("aggregate_vals argument must be one of None, average, max; you provided:"+aggregator)

    
    def on_epoch_end(self):
        if self.shuffle==True:
            #shuffle the indices!
            numrows=self.upsampled_indices.shape[0]
            df_indices=list(range(numrows))
            shuffle(df_indices)#this is an in-place operation
            df_indices=pd.Series(df_indices)

