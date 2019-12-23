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
from ..util import *
from .tiledb_generator import * 
import tiledb
import pdb

class TiledbPredictGenerator(TiledbGenerator):
    def __init__(self,
                 ref_fasta,
                 batch_size,
                 tdb_indexer,
                 tdb_partition_attribute_for_upsample,
                 tdb_partition_thresh_for_upsample,
                 upsample_ratio,
                 tdb_inputs,
                 tdb_input_source_attribute,
                 tdb_input_flank,
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
                 pseudocount=0,
                 expand_dims=False,
                 tiledb_stride=1,
                 bed_regions=None):
        
        TiledbGenerator.__init__(self,          
                                 ref_fasta=ref_fasta,
                                 batch_size=batch_size,
                                 tdb_indexer=tdb_indexer,
                                 tdb_partition_attribute_for_upsample=tdb_partition_attribute_for_upsample,
                                 tdb_partition_thresh_for_upsample=tdb_partition_thresh_for_upsample,
                                 upsample_ratio=upsample_ratio,
                                 tdb_inputs=tdb_inputs,
                                 tdb_input_source_attribute=tdb_input_source_attribute,
                                 tdb_input_flank=tdb_input_flank,
                                 tdb_outputs=tdb_outputs,
                                 tdb_output_source_attribute=tdb_output_source_attribute,
                                 tdb_output_flank=tdb_output_flank,
                                 num_inputs=num_inputs,
                                 num_outputs=num_outputs,
                                 tdb_input_aggregation=tdb_input_aggregation,
                                 tdb_input_transformation=tdb_input_transformation,
                                 tdb_output_aggregation=tdb_output_aggregation,
                                 tdb_output_transformation=tdb_output_transformation,
                                 chroms=chroms,
                                 chrom_sizes=chrom_sizes,
                                 shuffle_epoch_start=False,
                                 shuffle_epoch_end=False,
                                 pseudocount=0,
                                 add_revcomp=False,
                                 expand_dims=expand_dims,
                                 return_coords=True)
        self.tiledb_stride=tiledb_stride
        self.bed_regions=bed_regions

    def get_upsampled_indices(self):
        #use pandas dataframes to store index,chrom,position for upsampled and non-upsampled values
        upsampled_chroms=None
        upsampled_indices=None
        self.chrom_edge_flank=max([max(self.tdb_input_flank),max(self.tdb_output_flank)])
        
        for chrom in self.chroms_to_use:
            upsampled_indices_chrom=None
            chrom_size=None
            for task in self.data_arrays['index']:
                print(task)
                with tiledb.DenseArray(self.data_arrays['index'][task][chrom], mode='r',ctx=tiledb.Ctx(config=self.config)) as cur_array:
                    cur_vals=cur_array[:][self.tdb_partition_attribute_for_upsample]
                    print("got values for cur task/chrom") 
                if chrom_size is None:
                    chrom_size=cur_vals.shape[0]
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
        return 


    def get_coords(self,idx):
        if len(self.upsampled_indices)>0:
            upsampled_batch_start=idx*self.upsampled_batch_size
            upsampled_batch_end=min([upsampled_batch_start+self.upsampled_batch_size,self.upsampled_indices.shape[0]])
            upsampled_batch_indices=self.upsampled_indices.loc[list(range(upsampled_batch_start,upsampled_batch_end))]
            coords=upsampled_batch_indices
        else:
            #get genome position
            startpos=idx*self.batch_size*self.tiledb_stride
            #map to chromosome & chromosome position
            cur_chrom,chrom_start_pos=self.transform_idx_to_chrom_idx(startpos)
            coords=pd.DataFrame(range(chrom_start_pos,chrom_start_pos+self.batch_size*self.tiledb_stride,self.tiledb_stride),columns=['pos'])
            coords['chrom']=cur_chrom
        return coords

    
    def __len__(self):
        if len(self.upsampled_indices) is 0: 
            return int(ceil(self.length/(self.batch_size*self.tiledb_stride)))
        else:
            return int(ceil(self.upsampled_indices.shape[0] /self.batch_size))

    def on_epoch_end(self):
        pass

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
    
