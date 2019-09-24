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
                 label_subset_attribute=None,
                 label_thresh=None,
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
        self.label_subset_attribute=label_subset_attribute
        self.label_thresh=label_thresh
        self.indices_to_score=None
        self.sequence_flank=sequence_flank
        self.chrom_edge_flank=max([self.label_flank,self.sequence_flank])
        self.chrom_sizes_file=chrom_sizes_file
        self.chrom_sizes,self.last_index_to_chrom,self.length=self.get_genome_size()

        if self.label_subset_attribute is not None:
            print("filtering indices to score")
            self.indices_to_score=self.get_indices_to_score()
        

    def get_indices_to_score(self):
        #use pandas dataframes to store index,chrom,position for upsampled and non-upsampled values
        to_score_chroms=None
        to_score_indices=None

        for chrom in self.data_arrays:
            to_score_indices_chrom=None
            chrom_size=None
            for task in self.data_arrays[chrom]:
                with tiledb.DenseArray(self.data_arrays[chrom][task], mode='r') as cur_array:
                    cur_vals=cur_array[:][self.label_subset_attribute]
                if chrom_size is None:
                    chrom_size=cur_vals.shape[0]
                print("got values for cur task/chrom") 
                to_score_indices_task_chrom=np.argwhere(cur_vals>=self.label_thresh)
                print("got to_score indices")
                if to_score_indices_chrom is None:
                    to_score_indices_chrom=to_score_indices_task_chrom
                else:
                    to_score_indices_chrom=np.union1d(to_score_indices_chrom,to_score_indices_task_chrom)
                print("performed task union")
                
            #make sure we dont' run off the edges of the chromosome!
            usampled_indices_chrom=to_score_indices_chrom[to_score_indices_chrom>self.chrom_edge_flank]
            to_score_indices_chrom=to_score_indices_chrom[to_score_indices_chrom<(self.chrom_sizes[chrom]-self.chrom_edge_flank)]
            print("got indices to upsample for chrom:"+str(chrom))            
            if to_score_chroms is None:
                to_score_chroms=[chrom]*to_score_indices_chrom.shape[0]
                to_score_indices=to_score_indices_chrom
            else:
                to_score_chroms=to_score_chroms+[chrom]*to_score_indices_chrom.shape[0]
                to_score_indices=np.concatenate((to_score_indices,to_score_indices_chrom),axis=0)
            print("appended chrom indices to master list") 

        to_score_indices=pd.DataFrame.from_dict({'chrom':to_score_chroms,
                                                  'pos':to_score_indices.squeeze()})

        return to_score_indices
        
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
        if self.indices_to_score is None: 
            return int(floor(self.length/(self.batch_size*self.stride)))
        else:
            return self.indices_to_score.shape[0] 

    def __getitem__(self,idx):
        self.ctx = tiledb.Ctx()
        self.ref=pysam.FastaFile(self.ref_fasta)
        if self.indices_to_score is None: 
            return self.get_batch(idx) 
        else:
            return self.get_batch_from_pos_subset(idx) 
    def on_epoch_end(self):
        pass

    def get_batch_from_pos_subset(self,idx):
        cur_batch=self.indices_to_score.iloc[idx:idx+self.batch_size]
        x_pos=[(row['chrom'],row['pos']-self.sequence_flank,row['pos']+self.sequence_flank) for index,row in cur_batch.iterrows()]
        y_pos=[(row['chrom'],row['pos']-self.label_flank,row['pos']+self.label_flank) for index,row  in cur_batch.iterrows()]
        #get the sequences
        X=self.get_seqs(x_pos)

        #get the labels 
        y=self.get_labels(y_pos)
        return X,y,x_pos,y_pos
        
    def get_batch(self,idx):
        #get genome position
        startpos=idx*self.batch_size*self.stride
        #map to chromosome & chromosome position
        cur_chrom,chrom_start_pos=self.transform_idx_to_chrom_idx(startpos)
        positions=range(chrom_start_pos,chrom_start_pos+self.batch_size*self.stride,self.stride)
        x_pos=[(cur_chrom,p-self.sequence_flank,p+self.sequence_flank) for p in positions]
        y_pos=[(cur_chrom,p-self.label_flank,p+self.label_flank) for p in positions]
        
        #get the sequences
        X=self.get_seqs(x_pos)
        #get the labels 
        y=self.get_labels(y_pos)
        return X,y,x_pos,y_pos

    
    def get_seqs(self,positions):
        seqs=[]
        for pos in positions:
            try:
                cur_chrom=pos[0]
                start_coord=pos[1]
                end_coord=pos[2]
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
    
    def get_labels(self,positions):
        '''
        extract the labels from tileDB 
        '''
        label_vector_len=1
        if self.label_aggregation in ['None', None]:
            label_vector_len=2*self.label_flank         
        labels=np.zeros((len(positions),label_vector_len,len(self.tasks)))
        batch_entry_index=0
        for cur_pos in positions:
            cur_chrom=cur_pos[0]
            cur_start=cur_pos[1]
            cur_end=cur_pos[2]
            if cur_start < 0:
                labels[batch_entry_index,:,:].fill(np.nan)
            elif cur_end>=self.chrom_sizes[cur_chrom]:
                labels[batch_entry_index,:,:].fill(np.nan)
            else:
                for task_index in range(len(self.tasks)):
                    task=self.tasks[task_index]
                    with tiledb.DenseArray(self.data_arrays[cur_chrom][task], mode='r',ctx=self.ctx) as cur_array:
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
    
