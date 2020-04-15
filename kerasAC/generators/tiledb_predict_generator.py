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
                 tdb_array,
                 tdb_partition_attribute_for_upsample,
                 tdb_partition_thresh_for_upsample,
                 upsample_ratio,
                 tdb_input_source_attribute,
                 tdb_input_flank,
                 tdb_output_source_attribute,
                 tdb_output_flank,
                 tdb_ambig_attribute,
                 num_inputs,
                 num_outputs,
                 tdb_input_aggregation=None,
                 tdb_input_transformation=None,
                 tdb_output_aggregation=None,
                 tdb_output_transformation=None,
                 tdb_bias_arrays=None,
                 tdb_bias_source_attribute=None,
                 tdb_bias_flank=None,
                 tdb_bias_aggregation=None,
                 tdb_bias_transformation=None,
                 chroms=None,
                 chrom_sizes=None,
                 task_indices=None,
                 tasks=None,
                 pseudocount=0,
                 expand_dims=False,
                 tiledb_stride=1,
                 bed_regions=None,
                 tdb_config=None,
                 tdb_ctx=None,
                 num_threads=1):
        
        TiledbGenerator.__init__(self,          
                                 ref_fasta=ref_fasta,
                                 batch_size=batch_size,
                                 tdb_array=tdb_array,
                                 tdb_partition_attribute_for_upsample=tdb_partition_attribute_for_upsample,
                                 tdb_partition_thresh_for_upsample=tdb_partition_thresh_for_upsample,
                                 upsample_ratio=upsample_ratio,
                                 tdb_input_source_attribute=tdb_input_source_attribute,
                                 tdb_input_flank=tdb_input_flank,
                                 tdb_output_source_attribute=tdb_output_source_attribute,
                                 tdb_output_flank=tdb_output_flank,
                                 num_inputs=num_inputs,
                                 num_outputs=num_outputs,
                                 tdb_input_aggregation=tdb_input_aggregation,
                                 tdb_input_transformation=tdb_input_transformation,
                                 tdb_output_aggregation=tdb_output_aggregation,
                                 tdb_output_transformation=tdb_output_transformation,
                                 tdb_ambig_attribute=tdb_ambig_attribute,
                                 tdb_bias_arrays=tdb_bias_arrays,
                                 tdb_bias_source_attribute=tdb_bias_source_attribute,
                                 tdb_bias_flank=tdb_bias_flank,
                                 tdb_bias_aggregation=tdb_bias_aggregation,
                                 tdb_bias_transformation=tdb_bias_transformation,                                 
                                 chroms=chroms,
                                 chrom_sizes=chrom_sizes,
                                 shuffle_epoch_start=False,
                                 shuffle_epoch_end=False,
                                 pseudocount=pseudocount,
                                 add_revcomp=False,
                                 expand_dims=expand_dims,
                                 return_coords=True,
                                 tdb_config=tdb_config,
                                 tdb_ctx=tdb_ctx,
                                 tasks=tasks,
                                 task_indices=task_indices,
                                 num_threads=num_threads)
        self.tiledb_stride=tiledb_stride
        self.bed_regions=bed_regions
        print("created predict generator")
        


    def get_tdb_indices_for_batch(self,idx):
        if len(self.upsampled_indices)>0:
            #use the upsampled indices 
            upsampled_batch_start=idx*self.upsampled_batch_size
            upsampled_batch_end=min([upsampled_batch_start+self.upsampled_batch_size,self.upsampled_indices.shape[0]])
            upsampled_batch_indices=self.upsampled_indices[upsampled_batch_start:upsampled_batch_end]
            return upsampled_batch_indices
        else:
            #not upsampling, going through the test chromosomes with specified stride value 
            startpos=idx*self.batch_size*self.tiledb_stride
            batch_indices=[i for i in range(startpos,startpos+self.batch_size*self.tiledb_stride,self.tiledb_stride)]
            return batch_indices
    
    def __len__(self):
        if len(self.upsampled_indices) is 0: 
            return int(ceil(self.length/(self.batch_size*self.tiledb_stride)))
        else:
            return int(ceil(self.upsampled_indices.shape[0] /self.batch_size))

    def on_epoch_end(self):
        pass

