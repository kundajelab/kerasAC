from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
import threading 
from random import shuffle 
import math
from math import ceil, floor
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
    chrom_sizes_subset_dict=dict() 
    genome_size=chrom_sizes_subset[1].sum()
    last_index_to_chrom=dict()
    last_index=0
    for index,row in chrom_sizes_subset.iterrows():
        chrom_name=row[0]
        chrom_size=row[1]
        chrom_sizes_subset_dict[chrom_name]=chrom_size 
        last_index+=chrom_size
        last_index_to_chrom[last_index]=[chrom_name,chrom_size]
    return chrom_sizes_subset_dict,last_index_to_chrom, genome_size

            

class TiledbGenerator(Sequence):
    def __init__(self,
                 batch_size,
                 task_file,
                 ref_fasta,
                 label_source,
                 label_flank,
                 label_aggregation,
                 sequence_flank,
                 partition_attribute_for_upsample,
                 shuffle_epoch_start=True,
                 shuffle_epoch_end=True,
                 label_transformer=None,
                 chrom_sizes=None,
                 chroms=None,
                 partition_thresh_for_upsample=1,
                 upsample_ratio=0,
                 revcomp=False,
                 pseudocount=0):
        '''
        partition_attribute_for_upsample -- attribute in tiledb array used for determining which bases to upsample (usu. 'idr_peak') 
        partition_thresh_for_upsample -- threshold for determinining samples to upsample (generally 1) 
        label_aggregation -- one of 'avg','max',None
        '''

        self.shuffle_epoch_start=shuffle_epoch_start
        self.shuffle_epoch_end=shuffle_epoch_end
        self.ref_fasta=ref_fasta
        self.batch_size=batch_size
        self.tasks=open(task_file,'r').read().strip().split('\n')
        if chroms is not None:
            self.chroms_to_use=chroms
        else: 
            self.chroms_to_use=[i.split()[0] for i in open(chrom_sizes,'r').read().strip().split('\n')]            
        self.data_arrays=self.open_tiledb_arrays_for_reading()
        self.label_source=label_source
        self.label_flank=label_flank
        self.label_aggregation=label_aggregation
        self.label_transformer=label_transformer
        self.pseudocount=pseudocount
        self.sequence_flank=sequence_flank
        self.chrom_edge_flank=max([self.label_flank,self.sequence_flank])
        self.partition_attribute_for_upsample=partition_attribute_for_upsample
        self.partition_thresh_for_upsample=partition_thresh_for_upsample
        self.upsample_ratio=upsample_ratio
        self.chrom_sizes,self.last_index_to_chrom,self.length=get_genome_size(chrom_sizes,self.chroms_to_use)
        if self.upsample_ratio > 0:
            self.upsampled_indices=self.get_upsampled_indices()
            self.upsampled_indices_len=len(self.upsampled_indices)
            num_pos_wraps=math.ceil(self.length/self.upsampled_indices_len)
            self.upsampled_indices=pd.concat([self.upsampled_indices]*num_pos_wraps, ignore_index=True)[0:self.length]
        else:
            self.upsampled_indices_len=0
            self.upsampled_indices=[]            
        self.revcomp=revcomp
        if self.revcomp==True:
            self.batch_size=int(math.floor(self.batch_size/2))            
        self.upsampled_batch_size=math.ceil(self.upsample_ratio*self.batch_size)
        self.non_upsampled_batch_size=self.batch_size-self.upsampled_batch_size
        
    def open_tiledb_arrays_for_reading(self):
        '''
        Opens tiledb arrays for each task/chromosome for reading  
        '''
        array_dict=dict()
        for chrom in self.chroms_to_use:
            array_dict[chrom]=dict()
            for task in self.tasks:
                array_dict[chrom][task]= task+'.'+chrom
        return array_dict


    def get_upsampled_indices(self):
        #use pandas dataframes to store index,chrom,position for upsampled and non-upsampled values
        upsampled_chroms=None
        upsampled_indices=None

        for chrom in self.data_arrays:
            upsampled_indices_chrom=None
            chrom_size=None
            for task in self.data_arrays[chrom]:
                with tiledb.DenseArray(self.data_arrays[chrom][task], mode='r') as cur_array:
                    cur_vals=cur_array[:][self.partition_attribute_for_upsample]
                if chrom_size is None:
                    chrom_size=cur_vals.shape[0]
                print("got values for cur task/chrom") 
                upsampled_indices_task_chrom=np.argwhere(cur_vals>=self.partition_thresh_for_upsample)
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

        print("finished generator init")
        return upsampled_indices

        
    def __len__(self):
        return int(floor(self.length/self.batch_size))
    

    def __getitem__(self,idx):
        self.ref=pysam.FastaFile(self.ref_fasta)
        return self.get_batch(idx)
        
    def on_epoch_end(self):
        if self.shuffle==True:
            #shuffle the indices!
            numrows=self.upsampled_indices.shape[0]
            df_indices=list(range(numrows))
            shuffle(df_indices)#this is an in-place operation
            df_indices=pd.Series(df_indices)
            self.upsampled_indices=self.upsampled_indices.set_index(df_indices)
            
    def get_batch(self,idx):
        upsampled_batch_start=idx*self.upsampled_batch_size
        upsampled_batch_end=upsampled_batch_start+self.upsampled_batch_size
        X_upsampled=None
        X_non_upsampled=None
        
        if self.upsampled_batch_size > 0:
            upsampled_batch_indices=self.upsampled_indices.loc[list(range(upsampled_batch_start,upsampled_batch_end))]
            #get the sequences
            X_upsampled=self.get_seqs(upsampled_batch_indices)
            #get the labels 
            y_upsampled=self.get_labels(upsampled_batch_indices,self.upsampled_batch_size)
            
        if self.non_upsampled_batch_size > 0:
            #select random indices from genome
            non_upsampled_batch_indices=self.get_nonupsample_batch_indices()
            X_non_upsampled=self.get_seqs(non_upsampled_batch_indices)
            y_non_upsampled=self.get_labels(non_upsampled_batch_indices,self.non_upsampled_batch_size)

        #combine upsampled & non_upsampled batches
        if ((X_upsampled is not None) and (X_non_upsampled is not None)):
            X=np.concatenate((X_upsampled, X_non_upsampled),axis=0)
            y=np.concatenate((y_upsampled,y_non_upsampled),axis=0)
        elif X_upsampled is not None:
            X=X_upsampled
            y=y_upsampled
        else:
            X=X_non_upsampled
            y=y_non_upsampled
        return X,y
     
    def get_seqs(self,indices):
        seqs=[]
        for index,row in indices.iterrows():
            try:
                start_coord=row['pos']-self.sequence_flank
                end_coord=row['pos']+self.sequence_flank
                cur_chrom=row['chrom']
                if start_coord < 0:
                    raise Exception("start coordinate for sequence is < 0")
                if end_coord>=self.chrom_sizes[cur_chrom]:
                    raise Exception("end coordiante for sequence ("+str(end_coord)+") is greater than the size of chromosome:"+str(cur_chrom))
                seqs.append(self.ref.fetch(cur_chrom,start_coord,end_coord))
            except:
                #we are off the chromosome edge, just use all N's for the sequene in this edge case 
                seqs.append("N"*2*self.sequence_flank)
        if self.revcomp==True:
            seqs=seqs+revcomp(seqs)
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        return seqs

    def transform_label_vals(self,labels):
        self.label_transformer=str(self.label_transformer)
        if self.label_transformer == 'None':
            return labels
        elif self.label_transformer == 'asinh':
            return np.asinh(labels)
        elif self.label_transformer == 'log10':
            return np.log10(labels+self.pseudocount)
        elif self.label_transformer == 'log':
            return np.log(labels+self.pseudocount)
        else:
            raise Exception("transform_label_vals argument must be one of None, asinh, log10, log; you provided:"+str(self.label_transformer)) 
    
    def aggregate_label_vals(self,labels):
        self.label_aggregation=str(self.label_aggregation)
        if self.label_aggregation == 'None':
            return labels
        elif self.label_aggregation == 'average':
            return np.mean(labels)
        elif self.label_aggregation == 'max':
            return np.max(labels)
        else:
            raise Exception("label_aggregation argument must be one of None, average, max; you provided:"+str(self.label_aggregation))
    
    def get_labels(self,indices,batch_size):
        '''
        extract the labels from tileDB 
        '''
        #double the batch size implicitly if reverse-complemented inputs are being used for training 
        label_vector_len=1
        if self.label_aggregation == "None":
            label_vector_len=2*self.label_flank 
        
        labels=np.zeros((batch_size,label_vector_len,len(self.tasks)))
        batch_entry_index=0
        for index,row in indices.iterrows():
            cur_chrom=row['chrom']
            cur_pos=row['pos']
            cur_start=cur_pos-self.label_flank
            cur_end=cur_pos+self.label_flank
            for task_index in range(len(self.tasks)):
                task=self.tasks[task_index]
                ctx = tiledb.Ctx()
                with tiledb.DenseArray(self.data_arrays[cur_chrom][task], mode='r',ctx=ctx) as cur_array:
                    cur_vals=cur_array[cur_start:cur_end][self.label_source]
                vals=self.aggregate_label_vals(self.transform_label_vals(cur_vals))                    
                labels[batch_entry_index,:,task_index]=vals
            batch_entry_index+=1
        if self.revcomp==True:
            labels=np.concatenate((labels,labels),axis=0)
        return labels 

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
                    cur_chrom,cur_chrom_size=self.last_index_to_chrom[chrom_last_index]
                    cur_chrom_pos=random.randint(self.chrom_edge_flank, cur_chrom_size-self.chrom_edge_flank)
                    chroms.append(cur_chrom)
                    chrom_pos.append(cur_chrom_pos)
                    break 
        cur_batch= pd.DataFrame({'chrom':chroms,'pos':chrom_pos})
        assert cur_batch.shape[0]==self.non_upsampled_batch_size
        return cur_batch
