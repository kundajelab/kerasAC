from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

from multiprocessing.pool import ThreadPool 
from multiprocessing import Queue

#numpy & i/o
import warnings
import numpy as np
import argparse
import pysam
import pandas as pd

#import keras functions
import keras 
from keras import callbacks as cbks
from keras.losses import *
from keras.models import Model 

#import dependencies from locusselect
from ..metrics import *
from ..generators import *
from ..custom_losses import *
from ..config import args_object_from_args_dict

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model yaml & weights files & a dataset, get model predictions and accuracy metrics')
    parser.add_argument("--threads",type=int,default=1) 
    parser.add_argument("--max_queue_size",type=int,default=100)
    parser.add_argument('--model_hdf5',help='hdf5 file that stores the model')
    parser.add_argument("--input_grad",default=False, action="store_true",help="interpret with gradient x input") 
    parser.add_argument("--interpretation_layer",type=int,help="-1 for regression, -2 for classification")
    parser.add_argument("--deeplift_reference",help="one of 'shuffled_ref','gc_ref','zero_ref'")
    parser.add_argument("--deeplift_num_refs_per_seq",type=int,default=10,help="number of reference sequences to use for each sequence to be deepLIFTed") 
    parser.add_argument("--task_index",type=int,default=0,help="If the model is multi-tasked, select the index of the task to compute deeplift scores for; use 0 for single-tasked models")
    parser.add_argument('--input_bed_file',required=True,help='bed file with peaks')
    parser.add_argument('--preparsed_inputs',nargs="+",required=False,default=None,help='prepared inputs')
    parser.add_argument('--input_index_to_interpret',type=int,default=None)
    parser.add_argument('--batch_size',type=int,help='batch size to use to compute deepLIFT scores',default=1000)
    parser.add_argument('--ref_fasta')
    parser.add_argument('--flank',default=500,type=int)
    parser.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file) and expanded args.flank to the left and right")
    parser.add_argument("--center_on_bed_interval",default=False,action='store_true',help="if set to true, the region will be centered on the bed peak's center.")
    parser.add_argument("--output_npz_file",default=None,help="name of output file to store the deepLIFT scores. The npz file will have fields \"bed_entries\" and \"deeplift scores\"")
    parser.add_argument("--expand_dims",default=False,action="store_true",help="set to True if using 2D convolutions, Fales if 1D convolutions (default)")
    parser.add_argument("--sequential",default=True,action="store_true",help="set to True if model is sequtial (default), False if model is functional")
    return parser.parse_args()

import deeplift
import numpy as np

#Careful! Gradientxinput is summed across tasks, there is no support in tensorflow for calculating the per-task gradient
#(see thread here: https://github.com/tensorflow/tensorflow/issues/4897) 

def input_grad(model,X,target_layer_idx=-2,input_to_use=0):
    print("WARNING: this function provides aggregated gradients across tasks. Not recommended for multi-tasked models")
    from keras import backend as K 
    fn = K.function(model.inputs, K.gradients(model.layers[target_layer_idx].output, model.inputs))    
    return fn(X)[input_to_use]

def deeplift_zero_ref(X,score_func,batch_size=200,task_idx=0):        
    # use a 40% GC reference
    input_references = [np.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]]
    # get deeplift scores
    
    deeplift_scores = score_func(
        task_idx=task_idx,
        input_data_list=[X],
        batch_size=batch_size,
        progress_update=None,
        input_references_list=input_references)
    return deeplift_scores



def deeplift_gc_ref(X,score_func,batch_size=200,task_idx=0):        
    # use a 40% GC reference
    input_references = [np.array([0.3, 0.2, 0.2, 0.3])[None, None, None, :]]
    # get deeplift scores
    
    deeplift_scores = score_func(
        task_idx=task_idx,
        input_data_list=[X],
        batch_size=batch_size,
        progress_update=None,
        input_references_list=input_references)
    return deeplift_scores

def deeplift_shuffled_ref(X,score_func,batch_size=200,task_idx=0,num_refs_per_seq=10):
    deeplift_scores=score_func(task_idx=task_idx,input_data_sequences=X,num_refs_per_seq=num_refs_per_seq,batch_size=batch_size)
    return deeplift_scores

def get_deeplift_scoring_function(model,target_layer_idx=-2,task_idx=0, num_refs_per_seq=10,reference="shuffled_ref", sequential=True):
    """
    Arguments: 
        model -- a string containing the path to the hdf5 exported model 
        target_layer_idx -- should be -2 for classification; -1 for regression 
        reference -- one of 'shuffled_ref','gc_ref','zero_ref'
    Returns:
        deepLIFT scoring function 
    """
    from deeplift.conversion import kerasapi_conversion as kc
    deeplift_model = kc.convert_model_from_saved_files(model,verbose=False)

    #get the deeplift score with respect to the logit 
    if(sequential):
        score_func = deeplift_model.get_target_contribs_func(
             find_scores_layer_idx=task_idx,
             target_layer_idx=target_layer_idx)
    else:
        input_name = deeplift_model.get_input_layer_names()[0]
        target_layer_name = list(deeplift_model.get_name_to_layer().keys())[target_layer_idx]
        multipliers_func = deeplift_model.get_target_multipliers_func(input_name, target_layer_name)
        score_func = deeplift.util.get_hypothetical_contribs_func_onehot(multipliers_func)
    
    if reference=="shuffled_ref":
        from deeplift.util import get_shuffle_seq_ref_function
        from deeplift.dinuc_shuffle import dinuc_shuffle        
        score_func=get_shuffle_seq_ref_function(
            score_computation_function=score_func,
            shuffle_func=dinuc_shuffle,
            one_hot_func=None)
    return score_func


def deeplift_batch(score_func,X,task_idx,num_refs_per_seq,reference,batch_size):
    if reference=="shuffled_ref":
        deeplift_scores_batch=deeplift_shuffled_ref(X,score_func,batch_size,task_idx,num_refs_per_seq)
    elif reference=="gc_ref":
        deeplift_scores_batch=deeplift_gc_ref(X,score_func,batch_size,task_idx)
    elif reference=="zero_ref":
        deeplift_scores_batch=deeplift_zero_ref(X,score_func,batch_size,task_idx)
    else:
        raise Exception("supported DeepLIFT references are 'shuffled_ref','gc_ref', 'zero_ref'")
    print("done with batch")
    #Project onto the base that's actually present
    deeplift_scores_batch = np.sum(deeplift_scores_batch, axis=-1)[:,:,:,None]*X
    return deeplift_scores_batch

def batch_extract(inputs):
    batch_index=inputs[0]
    data_generator=inputs[1]
    X,bed_entries=data_generator[batch_index]
    return X,bed_entries,batch_index

def load_model(model_string):
    from keras.models import load_model    
    custom_objects={"recall":recall,
                    "sensitivity":recall,
                    "specificity":specificity,
                    "fpr":fpr,
                    "fnr":fnr,
                    "fdr":fdr,
                    "precision":precision,
                    "f1":f1,
                    "ambig_binary_crossentropy":ambig_binary_crossentropy,
                    "ambig_mean_squared_error":ambig_mean_squared_error}    
    model=load_model(model_string,custom_objects=custom_objects)
    return model



def compute_interpretation_scores(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    
    #create generator to score batches of deepLIFT data
    data_generator=DataGenerator(args.input_bed_file,
                                 args.ref_fasta,
                                 batch_size=args.batch_size,
                                 center_on_summit=args.center_on_summit,
                                 center_on_bed_interval=args.center_on_bed_interval,
                                 flank=args.flank,
                                 expand_dims=args.expand_dims)
    print("created data generator")
    if args.input_grad is False:
        #make sure an appropriate reference was specified
        assert args.deeplift_reference in ["shuffled_ref","gc_ref","zero_ref"]
        
        #get the deepLIFT scoring function
        score_func=get_deeplift_scoring_function(args.model_hdf5,
                                             target_layer_idx=args.interpretation_layer,
                                             task_idx=args.task_index,
                                             num_refs_per_seq=args.deeplift_num_refs_per_seq,
                                             reference=args.deeplift_reference,
                                             sequential=args.sequential)
    else:
        #load the model
        model=load_model(args.model_hdf5)
    scores=None
    bed_entries=None
    pool_inputs=[]
    len_data_generator=len(data_generator)
    print(len_data_generator)
    processed=[] 
    for batch_index in range(len(data_generator)+1):
        pool_inputs.append([batch_index,data_generator])
    pool=ThreadPool(args.threads)
    for batch in pool.imap(batch_extract, pool_inputs):
        X=batch[0]
        bed_entries_batch=batch[1]
        batch_id=batch[2]
        print(batch_id) 
        processed.append(batch_id)
        if len(processed)==(len_data_generator+1):
            break
        if args.input_grad is True:
            batch_scores=input_grad(model,X,target_layer_idx=args.interpretation_layer,input_to_use=args.input_index_to_interpret)
        else:
            batch_scores=deeplift_batch(score_func,X,args.task_index,args.deeplift_num_refs_per_seq,args.deeplift_reference,args.batch_size)
        if scores is None:
            scores=batch_scores
            bed_entries=bed_entries_batch
        else:
            scores=np.append(scores,batch_scores,axis=0)
            bed_entries=np.append(bed_entries,bed_entries_batch,axis=0)
    if args.output_npz_file is not None:
        print("writing output file")
        np.savez_compressed(args.output_npz_file,bed_entries=bed_entries,interp_scores=scores)
    pool.close() 
    pool.join()
    return bed_entries,scores    
    
def main():
    args=parse_args()
    compute_interpretation_scores(args) 

    

if __name__=="__main__":
    main()
    
