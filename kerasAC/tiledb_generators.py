from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
import math 
import pysam
from .util import *
import tiledb
import pdb 

def get_upsampled_indices(data_arrays,
                          partition_attribute_for_upsample,
                          partition_thresh_for_upsample,
                          shuffle):
    #use pandas dataframes to store index,chrom,position for upsampled and non-upsampled values 
    upsampled_indices=None
    non_upsampled_indices=None

    for chrom in data_arrays:
        upsampled_indices_chrom=None
        non_upsampled_indices_chrom=None
        chrom_size=None
        for task in data_arrays[chrom]:
            cur_vals=data_arrays[chrom][task][:][partition_attribute_for_upsample]
            if chrom_size is None:
                chrom_size=cur_vals.shape[0]
            print("got values for cur task/chrom") 
            upsampled_indices_task_chrom=np.argwhere(cur_vals>=partition_thresh_for_upsample)
            print(upsampled_indices_task_chrom.shape)
            if upsampled_indices_chrom is None:
                upsampled_indices_chrom=upsampled_indices_task_chrom
            else:
                upsampled_indices_chrom=np.union1d(upsampled_indices_chrom,upsampled_indices_task_chrom)
        print("got indices to upsample for chrom:"+str(chrom))

        mask = np.zeros(chrom_size, dtype=bool)
        mask[upsampled_indices_chrom] = True
        non_upsampled_indices_chrom = np.array(range(chrom_size))[~mask]
        print("got indices to NOT upsampled for chrom:"+str(chrom))

        upsampled_chrom_name_array=[chrom]*upsampled_indices_chrom.shape[0]
        non_upsampled_chrom_name_array=[chrom]*non_upsampled_indices_chrom.shape[0]

        cur_upsampled_df=pd.DataFrame.from_dict({'chrom':upsampled_chrom_name_array,
                                               'indices':upsampled_indices_chrom.flatten()})
        cur_non_upsampled_df=pd.DataFrame.from_dict({'chrom':non_upsampled_chrom_name_array,
                                                   'indices':non_upsampled_indices_chrom.flatten()})
        print("generated coord dataframes for chrom:"+str(chrom))
        if upsampled_indices is None:
            upsampled_indices=cur_upsampled_df
            non_upsampled_indices=cur_non_upsampled_df
        else:
            upsampled_indices=pd.concat([upsampled_indices,cur_upsampled_df],axis=0)
            non_upsampled_indices=pd.concat([non_upsampled_indices,cur_non_upsampled_df],axis=0)
        print("added chrom coords to master list")

    if shuffle==True:
        print("shuffling upsampled and non-upsampled dataframes prior to start of training")
        upsampled_indices.apply(np.random.shuffle,axis=0)
        non_upsampled_indices.apply(np.random.shuffle,axis=0)

    print("finished generator init")
    return upsampled_indices,non_upsampled_indices 


def get_upsampled_indices_old(data_arrays,
                          partition_attribute_for_upsample,
                          partition_thresh_for_upsample,
                          shuffle):
    upsampled_indices=[]
    non_upsampled_indices=[]
    
    for chrom in data_arrays:
        positions=None
        for task in data_arrays[chrom]:
            cur_vals=data_arrays[chrom][task][:][partition_attribute_for_upsample]
            if positions is None:
                positions=np.zeros(cur_vals.shape)
            print("got values for cur task/chrom") 
            upsampled_indices_task_chrom=np.argwhere(cur_vals>=partition_thresh_for_upsample)
            print("got indices to upsample for task/chrom") 
            positions[upsampled_indices_task_chrom]=1
        print("aggregating indices to upsample across chroms") 
        for index in np.argwhere(positions==1):
            upsampled_indices.append((chrom,index))
        print("aggregated upsampled")
        for index in np.argwhere(positions==0):
            non_upsampled_indices.append((chrom,index))
        print("done with chrom:"+str(chrom))
    if shuffle==True:
        random.shuffle(upsampled_indices)
        random.shuffle(non_upsampled_indices)
    return upsampled_indices,non_upsampled_indices 

            
def open_tiledb_arrays_for_reading(tasks,chroms):
    '''
    Opens tiledb arrays for each task/chromosome for reading  
    '''
    array_dict=dict()
    for chrom in chroms:
        array_dict[chrom]=dict()
        for task in tasks:
            array_dict[chrom][task]= tiledb.DenseArray(task+'.'+chrom,mode='r')
    print("opened array dict") 
    return array_dict

class TiledbGenerator(Sequence):
    def __init__(self,
                 chrom_sizes=None,
                 chroms=None,
                 shuffle_epoch_start,
                 shuffle_epoch_end,
                 batch_size,
                 task_file,
                 label_source,
                 label_flank,
                 label_aggregation,
                 sequence_flank,
                 partition_attribute_for_upsample,
                 partition_thresh_for_upsample=1,
                 upsample_ratio=0,
                 revcomp=False,
                 transform_label_vals=None,
                 pseudocount=0):
        '''
        partition_attribute_for_upsample -- attribute in tiledb array used for determining which bases to upsample (usu. 'idr_peak') 
        partition_thresh_for_upsample -- threshold for determinining samples to upsample (generally 1) 
        label_aggregation -- one of 'avg','max',None
        '''
        self.shuffle_epoch_start=shuffle_epoch_start
        self.shuffle_epoch_end=shuffle_epoch_end
        self.batch_size=batch_size
        self.tasks=open(task_file,'r').read().strip().split('\n')
        if chroms is not None:
            self.chroms_to_use=chroms
        else: 
            self.chroms_to_use=[i.split()[0] for i in open(chrom_sizes,'r').read().strip().split('\n')]
            
        self.data_arrays=open_tiledb_arrays_for_reading(self.tasks,self.chroms_to_use)
        self.label_source=label_source
        self.label_flank=label_flank
        self.label_aggregation=label_aggregation
        self.transform_label_vals=transform_label_vals
        self.pseudocount=pseudocount
        self.sequence_flank=sequence_flank
        self.partition_attribute_for_upsample=partition_attribute_for_upsample
        self.partition_thresh_for_upsample=partition_thresh_for_upsample
        self.upsampled_indices, self.non_upsampled_indices=get_upsampled_indices(self.data_arrays,
                                                                                 self.partition_attribute_for_upsample,
                                                                                 self.partition_thresh_for_upsample,
                                                                                 self.shuffle)
        self.upsampled_indices_len=len(self.upsampled_indices)
        self.non_upsampled_indices_len=len(self.non_upsampled_indices)
        self.revcomp=revcomp
        if self.revcomp==True:
            self.batch_size=int(math.floor(self.batch_size/2))
            
        self.upsampled_batch_size=math.ceil(self.upsample_ratio*self.batch_size)
        self.non_upsampled_batch_size=self.batch_size-self.upsampled_batch_size
        
        
        
    def __len__(self):
        return math.ceil(max([self.upsampled_indices_len,
                              self.non_upsampled_indices_len])/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            self.ref=pysam.FastaFile(self.ref_fasta)
            return self.get_batch(idx)
        
    def on_epoch_end(self):
        if self.shuffle==True:
            random.shuffle(self.upsampled_indices)
            random.shuffle(self.non_upsampled_indices)
            
    def get_batch(self,idx):
        upsampled_idx=idx % self.upsampled_indices_len
        non_upsampled_idx=idx % self.non_upsampled_indices_len 
        upsampled_indices=self.upsampled_indices[upsampled_idx*self.upsampled_batch_size:(upsampled_idx+1)*self.upsampled_batch_size]
        non_upsampled_indices=self.non_upsampled_indices[non_upsampled_idx*self.non_upsampled_batch_size:(non_upsampled_idx+1)*self.non_upsampled_batch_size]

        #get the sequences
        X_upsampled=self.get_seqs(upsampled_indices)
        X_non_upsampled=self.get_seqs(non_upsampled_indices)
        X=np.concatenate((X_upsampled, X_non_upsampled),axis=0)
        
        #get the labels
        y_upsampled=self.get_labels(upsampled_indices,self.upsampled_batch_size)
        y_non_upsampled=self.get_labels(non_upsampled_indices,self.non_upsampled_batch_size) 
        y=np.concatenate((y_upsampled,y_non_upsampled),axis=0)
        return X,y
    
    def get_seqs(self,indices):
        seqs=[self.ref.fetch(i[0],i[1]-self.sequence_flank,i[1]+self.sequence_flank-1) for i in indices]
        if self.revcomp==True:
            seqs=seqs+revcomp(seqs)
        onehot_seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        return seqs

    def transform_label_vals(self,labels):
        if self.transform_label_vals is None:
            return labels
        elif self.transform_label_vals is'asinh':
            return np.asinh(labels)
        elif self.transform_label_vals is 'log10':
            return np.log10(labels+self.pseudocount)
        elif self.transform_label_vals is 'log':
            return np.log(labels+self.pseudocount)
        else:
            raise Exception("transform_label_vals argument must be one of None, asinh, log10, log; you provided:"+str(self.transform_label_vals)) 
    
    def aggregate_label_vals(self,labels):
        if self.label_aggregation is None:
            return labels
        elif self.label_aggregation is 'average':
            return np.mean(labels)
        elif self.label_aggregation is 'max':
            return np.max(labels)
        else:
            raise Exception("label_aggregation argument must be one of None, average, max; you provided:"+str(self.label_aggregation))
    
    
    def get_labels(self,indices,batch_size):
        '''
        extract the labels from tileDB 
        '''
        #double the batch size implicitly if reverse-complemented inputs are being used for training 
            
        label_vector_len=1
        if self.label_aggregation is None:
            label_vector_len=2*self.label_flank 
        
        labels=np.zeros((batch_size,label_vector_len,len(self.tasks)))
        for i in range(len(indices)):
            index=indices[i]
            cur_chrom=index[0]
            cur_start=index[1]-self.label_flank
            cur_end=index[1]+self.label_flank-1
            for task_index in len(self.tasks):
                task=self.tasks[task_index]
                vals=self.aggregate_label_vals(
                    self.transform_label_vals(
                        self.array_dict[cur_chrom][task][cur_start:cur_end][self.label_flank])
                    )
                labels[i][:][task_index]=vals
        if self.revcomp==True:
            labels=np.concatenate((labels,labels),axis=0)
        return labels 

