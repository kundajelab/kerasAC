from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
from random import shuffle 
import math 
import pysam
from .util import *
import tiledb
import pdb

def get_genome_size(chrom_sizes_file,chroms):
    '''
    get size of chromosomes to train on 
    '''
    chrom_sizes=pd.read_csv(chrom_sizes_file,header=None,sep='\t')
    chrom_sizes_subset=chrom_sizes[chrom_sizes[0].isin(chroms)]
    genome_size=chrom_sizes_subset[1].sum()
    last_index_to_chrom=dict()
    last_index=0
    for index,row in chrom_sizes_subset.iterrows():
        chrom_name=row[0]
        chrom_size=row[1]
        last_index+=chrom_size
        last_index_to_chrom[last_index]=[chrom_name,chrom_size]
    return last_index_to_chrom, genome_size

def get_nonupsample_batch_indices(n,last_index_to_chrom,length):
    '''
    randomly select n positions from the genome 
    '''
    indices=random.sample(range(length),n)
    #get the chroms and coords for each index
    chroms=[]
    chrom_pos=[]
    for cur_index in indices:
        for chrom_last_index in last_index_to_chrom:
            if cur_index < chrom_last_index:
                #this is the chromosome to use!
                cur_chrom,cur_chrom_size=last_index_to_chrom[chrom_last_index]
                cur_chrom_pos=cur_index % cur_chrom_size
                chroms.append(cur_chrom)
                chrom_pos.append(cur_chrom_pos)
    return pd.DataFrame({'chrom':chroms,'pos':chrom_pos})

def get_upsampled_indices(data_arrays,
                          partition_attribute_for_upsample,
                          partition_thresh_for_upsample,
                          shuffle_epoch_start,
                          upsample_ratio):
    #use pandas dataframes to store index,chrom,position for upsampled and non-upsampled values
    upsampled_chroms=None
    upsampled_indices=None

    for chrom in data_arrays:
        upsampled_indices_chrom=None
        chrom_size=None
        for task in data_arrays[chrom]:
            print("starting:"+str(task)+":"+str(chrom))
            cur_vals=data_arrays[chrom][task][:][partition_attribute_for_upsample]
            print(cur_vals.shape)
            if chrom_size is None:
                chrom_size=cur_vals.shape[0]
            print("got values for cur task/chrom") 
            upsampled_indices_task_chrom=np.argwhere(cur_vals>=partition_thresh_for_upsample)
            print("got upsampled indices")
            if upsampled_indices_chrom is None:
                upsampled_indices_chrom=upsampled_indices_task_chrom
            else:
                upsampled_indices_chrom=np.union1d(upsampled_indices_chrom,upsampled_indices_task_chrom)
            print("performed task union")
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
    if shuffle_epoch_start==True:
        numrows=upsampled_indices.shape[0]
        df_indices=list(range(numrows))
        shuffle(df_indices)#this is an in-place operation
        df_indices=pd.Series(df_indices)
        upsampled_indices=upsampled_indices.set_index(df_indices)
        print("shuffling upsampled dataframes prior to start of training")
        #print(upsampled_indices.head())

    print("finished generator init")
    print(upsampled_indices.head())
    return upsampled_indices

            
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
                 shuffle_epoch_start,
                 shuffle_epoch_end,
                 batch_size,
                 task_file,
                 label_source,
                 label_flank,
                 label_aggregation,
                 sequence_flank,
                 partition_attribute_for_upsample,
                 chrom_sizes=None,
                 chroms=None,
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
        self.upsample_ratio=upsample_ratio

        self.upsampled_indices=get_upsampled_indices(self.data_arrays,
                                                      self.partition_attribute_for_upsample,
                                                      self.partition_thresh_for_upsample,
                                                      self.shuffle_epoch_start,
                                                      self.upsample_ratio)
        self.upsampled_indices_len=len(self.upsampled_indices)
        self.last_index_to_chrom,self.length=get_genome_size(chrom_sizes,self.chroms_to_use)
        self.revcomp=revcomp
        if self.revcomp==True:
            self.batch_size=int(math.floor(self.batch_size/2))
            
        self.upsampled_batch_size=math.ceil(self.upsample_ratio*self.batch_size)
        self.non_upsampled_batch_size=self.batch_size-self.upsampled_batch_size
        
    def __len__(self):
        return self.length/self.batch_size
    

    def __getitem__(self,idx):
        with self.lock:
            self.ref=pysam.FastaFile(self.ref_fasta)
            return self.get_batch(idx)
        
    def on_epoch_end(self):
        if self.shuffle==True:
            random.shuffle(self.upsampled_indices)

    def get_batch(self,idx):
        upsampled_idx=idx % self.upsampled_indices_len
        non_upsampled_idx=idx % self.length
        upsampled_batch_indices=self.upsampled_indices.loc[upsampled_idx*self.upsampled_batch_size:(upsampled_idx+1)*self.upsampled_batch_size]
        #select random indices from genome 
        non_upsampled_batch_indices=get_nonupsample_batch_indices(self.non_upsampled_batch_size,self.last_index_to_chrom,self.length)

        #get the sequences
        X_upsampled=self.get_seqs(upsampled_batch_indices)
        X_non_upsampled=self.get_seqs(non_upsampled_batch_indices)
        X=np.concatenate((X_upsampled, X_non_upsampled),axis=0)
        
        #get the labels
        y_upsampled=self.get_labels(upsampled_indices,self.upsampled_batch_size)
        y_non_upsampled=self.get_labels(non_upsampled_indices,self.non_upsampled_batch_size) 
        y=np.concatenate((y_upsampled,y_non_upsampled),axis=0)
        return X,y
    
    def get_seqs(self,indices):
        pdb.set_trace() 
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

