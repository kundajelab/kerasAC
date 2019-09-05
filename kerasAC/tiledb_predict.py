#predictions from tiledb
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


            

class TiledbPredictGenerator(Sequence):
    def __init__(self,
                 batch_size,
                 task_file,
                 ref_fasta,
                 label_source,
                 label_flank,
                 label_aggregation,
                 sequence_flank,
                 tiledb_stride=1,
                 bed_regions=None,
                 label_transformer=None,
                 chrom_sizes_file=None,
                 chroms=None):
        '''
        Generator for getting sequences/labels out of tiledb 
        label_aggregation -- one of 'avg','max',None
        '''
        self.ref_fasta=ref_fasta
        self.batch_size=batch_size
        self.tasks=open(task_file,'r').read().strip().split('\n')
        if chroms is not None:
            self.chroms_to_use=chroms
        else: 
            self.chroms_to_use=[i.split()[0] for i in open(chrom_sizes,'r').read().strip().split('\n')]
        self.stride=tiledb_stride
        self.data_arrays=self.open_tiledb_arrays_for_reading()
        self.label_source=label_source
        self.label_flank=label_flank
        self.label_aggregation=label_aggregation
        self.label_transformer=label_transformer
        self.sequence_flank=sequence_flank
        self.chrom_edge_flank=max([self.label_flank,self.sequence_flank])
        self.chrom_sizes_file=chrom_sizes_file
        self.chrom_sizes,self.last_index_to_chrom,self.length=self.get_genome_size()
        
    def get_genome_size(self):
        '''
        get size of chromosomes to train on
        '''
        chrom_sizes=pd.read_csv(self.chrom_sizes_file,header=None,sep='\t')
        chrom_sizes_subset=chrom_sizes[chrom_sizes[0].isin(self.chroms_to_use)]
        chrom_sizes_subset_dict=dict() 
        genome_size=chrom_sizes_subset[1].sum()
        first_last_index_to_chrom=dict()
        last_index=0
        first_index=0 
        for index,row in chrom_sizes_subset.iterrows():
            chrom_name=row[0]
            chrom_size=row[1]
            chrom_sizes_subset_dict[chrom_name]=chrom_size 
            last_index+=chrom_size
            first_last_index_to_chrom[last_index]=[chrom_name,chrom_size,first_index]
            first_index+=chrom_size
        return chrom_sizes_subset_dict,first_last_index_to_chrom, genome_size
        
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

        
    def __len__(self):
        return int(floor(self.length/(self.batch_size*self.stride)))
    

    def __getitem__(self,idx):
        self.ref=pysam.FastaFile(self.ref_fasta)
        return self.get_batch(idx) 
        
    def on_epoch_end(self):
        pass
    
    def get_batch(self,idx):
        #get genome position
        startpos=idx*self.batch_size*self.stride
        #map to chromosome & chromosome position
        cur_chrom,chrom_start_pos=self.transform_idx_to_chrom_idx(startpos)
        positions=range(chrom_start_pos,chrom_start_pos+self.batch_size*self.stride,self.stride)
        #get the sequences
        X=self.get_seqs(cur_chrom,positions)
        #get the labels 
        y=self.get_labels(cur_chrom,positions)
        all_pos=[(cur_chrom,p-self.label_flank,p+self.label_flank) for p in positions]
        return X,y,all_pos

    
    def get_seqs(self,cur_chrom,positions):
        seqs=[]
        for pos in positions:
            try:
                start_coord=pos-self.sequence_flank
                end_coord=pos+self.sequence_flank
                if start_coord < 0:
                    raise Exception("start coordinate for sequence is < 0")
                if end_coord>=self.chrom_sizes[cur_chrom]:
                    raise Exception("end coordinate for sequence ("+str(end_coord)+") is greater than the size of chromosome:"+str(cur_chrom))                                
                seqs.append(self.ref.fetch(cur_chrom,start_coord,end_coord))
            except:
                #we are off the chromosome edge, just use all N's for the sequene in this edge case
                seqs.append("N"*2*self.sequence_flank)
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
    
    def get_labels(self,cur_chrom,positions):
        '''
        extract the labels from tileDB 
        '''
        label_vector_len=1
        if self.label_aggregation in ['None', None]:
            label_vector_len=2*self.label_flank         
        labels=np.zeros((self.batch_size,label_vector_len,len(self.tasks)))
        batch_entry_index=0
        for cur_pos in positions:
            cur_start=cur_pos-self.label_flank
            cur_end=cur_pos+self.label_flank
            if cur_start < 0:
                labels[batch_entry_index,:,:].fill(np.nan)
            elif cur_end>=self.chrom_sizes[cur_chrom]:
                labels[batch_entry_index,:,:].fill(np.nan)
            else:
                for task_index in range(len(self.tasks)):
                    task=self.tasks[task_index]
                    ctx = tiledb.Ctx()
                    with tiledb.DenseArray(self.data_arrays[cur_chrom][task], mode='r',ctx=ctx) as cur_array:
                        cur_vals=cur_array[cur_start:cur_end][self.label_source]
                    vals=self.aggregate_label_vals(self.transform_label_vals(cur_vals))
                    labels[batch_entry_index,:,task_index]=vals
            batch_entry_index+=1
        return labels 

    def transform_idx_to_chrom_idx(self,pos):
        '''
        transform the provided genome-level idx to a chrom-level idx 
        '''
        #get the chroms and coords for each index
        for chrom_last_index in self.last_index_to_chrom:
            if pos < chrom_last_index:
                #this is the chromosome to use!
                #make sure we don't slide off the edge of the chromosome 
                cur_chrom,cur_chrom_size,cur_chrom_first_index=self.last_index_to_chrom[chrom_last_index]
                cur_chrom_pos=pos-cur_chrom_first_index
                return cur_chrom, cur_chrom_pos
        raise Exception("invalid index -- larger than the genome size")
    
