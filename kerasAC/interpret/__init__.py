from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from multiprocessing.pool import Pool 

#numpy & i/o
import warnings
import numpy as np
import argparse
import pysam
import pandas as pd

#import dependencies from kerasAC 
from ..metrics import *
from ..generators.snp_generator import *
from ..generators.basic_generator import * 
from ..custom_losses import *
from ..config import args_object_from_args_dict
from ..get_model import * 
from .ism import *
from .deeplift import *
from .deepshap import *
from .input_grad import * 

interp_methods={'ism':ism_wrapper,
                'deeplift':deeplift_wrapper,
                'deepshap':deepshap_wrapper,
                'input_grad':input_grad_wrapper}

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model yaml & weights files & a dataset, get model predictions and accuracy metrics')

    parser.add_argument("--output_npz_file",default=None,help="name of output file to store the interpretation scores. The npz file will have fields \"bed_entries\" and \"scores\"")
    parser.add_argument("--generator_type", choices=['basic','snp'],help="snp uses snp_generator to interpret ref and alt alleles; basic uses basic_generator to interpret a sequence")
    parser.add_argument("--interp_method",choices=['ism','input_grad','deeplift','deepshap'])
    
    parallelization_group=parser.add_argument_group('parallelization')
    parallelization_group.add_argument("--threads",type=int,default=1) 
    parallelization_group.add_argument("--max_queue_size",type=int,default=100)

    model_group=parser.add_argument_group("model")
    model_group.add_argument('--load_model_hdf5',help='hdf5 file that stores the model')
    model_group.add_argument('--w1_w0_file',default=None)
    model_group.add_argument("--w1",default=None)
    model_group.add_argument("--w0",default=None)
    model_group.add_argument("--yaml",default=None)
    model_group.add_argument("--json",default=None)
    model_group.add_argument("--architecture_from_file",default=None)
    model_group.add_argument("--num_gpus",type=int,default=1)
    model_group.add_argument("--weights",default=None)
    
    model_group.add_argument("--expand_dims",default=False,action="store_true",help="set to True if using 2D convolutions, Fales if 1D convolutions (default)")
    model_group.add_argument('--batch_size',type=int,help='batch size to use to compute deepLIFT scores',default=1000)
    model_group.add_argument("--sequential",default=True,action="store_true",help="set to True if model is sequtial (default), False if model is functional")


    input_group=parser.add_argument_group("inputs")
    input_group.add_argument('--input_bed_file',required=True,help='bed file with peaks')
    input_group.add_argument('--ref_fasta')
    input_group.add_argument('--flank',default=500,type=int)
    input_group.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file) and expanded args.flank to the left and right")
    input_group.add_argument("--center_on_bed_interval",default=False,action='store_true',help="if set to true, the region will be centered on the bed peak's center.")
    input_group.add_argument("--compute_gc",action="store_true",default=False)
    
    snp_group=parser.add_argument_group('snp')
    snp_group.add_argument("--chrom_col")
    snp_group.add_argument("--pos_col")
    snp_group.add_argument("--ref_col")
    snp_group.add_argument("--alt_col")
    snp_group.add_argument("--rsid_col")
    
    
    interp_group=parser.add_argument_group("interp")
    interp_group.add_argument("--target_layer",type=int,help="-1 for regression, -2 for classification")
    interp_group.add_argument("--task_index",type=int,default=0,help="If the model is multi-tasked, select the index of the task to compute deeplift scores for; use 0 for single-tasked models")
    interp_group.add_argument('--input_index_to_interpret',type=int,default=0)
    interp_group.add_argument("--deepshap_reference",choices=['shuffled_ref'])
    interp_group.add_argument("--deepshap_num_refs_per_seq",type=int,default=10,help="number of reference sequences to use for each sequence to be deepSHAPed")
    interp_group.add_argument("--deeplift_reference",choices=['shuffled_ref','gc_ref','zero_ref'])
    interp_group.add_argument("--deeplift_num_refs_per_seq",type=int,default=10,help="number of reference sequences to use for each sequence to be deepLIFTed") 
    
    return parser.parse_args()

def get_generators(args):     
    if args.generator_type=="basic":    
        #create generator to score batches of deepLIFT data
        return [DataGenerator(args.input_bed_file,
                                     args.ref_fasta,
                                     batch_size=args.batch_size,
                                     center_on_summit=args.center_on_summit,
                                     center_on_bed_interval=args.center_on_bed_interval,
                                     flank=args.flank,
                                     expand_dims=args.expand_dims)],''
    elif args.generator_type=="snp":
        ref_generator=SNPGenerator(args.input_bed_file,
                                   args.chrom_col,
                                   args.pos_col,
                                   args.ref_col,
                                   args.flank,
                                   args.ref_fasta,
                                   rsid_col=args.rsid_col,
                                   compute_gc=args.compute_gc,
                                   batch_size=args.batch_size,
                                   expand_dims=args.expand_dims)
        alt_generator=SNPGenerator(args.input_bed_file,
                                   args.chrom_col,
                                   args.pos_col,
                                   args.alt_col,
                                   args.flank,
                                   args.ref_fasta,
                                   rsid_col=args.rsid_col,
                                   compute_gc=args.compute_gc,
                                   batch_size=args.batch_size,
                                   expand_dims=args.expand_dims)
        return [ref_generator,alt_generator],['ref','alt']
    else:
        raise Exception('unsupported value provided for generator_type argument; must be one of "snp" or "basic"')
    

def interpret(generator,model,args):
    print("starting interpretation...")
    scores=None
    bed_entries=None
    #get static inputs for interpreting each batch 
    static_inputs=[]
    if args.interp_method in ['ism','ism_gc']:
        preacts=get_preact_function(model,args.target_layer)
        static_inputs.append(preacts)
        static_inputs.append(args.task_index)
        static_inputs.append(args.compute_gc)
        print("generated static inputs for ism/ism_gc")
    elif args.interp_method in ['deeplift']:
        score_func=get_deeplift_scoring_function(model,
                                                 args.target_layer,
                                                 args.task_index,
                                                 reference=args.deeplift_reference,
                                                 sequential=args.sequential)
        static_inputs.append(score_func)
        static_inputs.append(args.task_layer_idx)
        static_inputs.append(args.deeplift_num_refs_per_seq)
        static_inputs.append(args.deeplift_reference)
        print("generated static inputs for deeplift")
    elif args.interp_method in ['deepshap']:
        if args.expand_dims==True:
            #are we using 2d or 1d convolutions? 
            combine_mult_and_diffref=combine_mult_and_diffref_2d
        else:
            combine_mult_and_diffref=combine_mult_and_diffref_1d
        explainer=create_explainer(model,
                                   create_background,
                                   args.target_layer,
                                   combine_mult_and_diffref,
                                   args.task_index)
        static_inputs.append(explainer)
        print("generated static inputs for deepshap") 
    elif args.interp_method in ['input_grad']:
        grad_function=get_input_grad_function(model,args.target_layer)
        static_inputs.append(grad_function)
        static_inputs.append(args.input_to_use)
        print("generated static inputs for input_grad")
    else:
        raise Exception('invalid interpretation method specified!')

    print("iterating...")
    for bed_entries_batch,X in generator:
        print(X[0].shape)
        print(len(X[1]))
        print("getting scores for batch!")
        batch_scores=interp_methods[args.interp_method]([X]+static_inputs)
    
        if scores is None:
            scores=batch_scores
            bed_entries=bed_entries_batch
        else:
            scores=np.append(scores,batch_scores,axis=0)
            bed_entries=np.append(bed_entries,bed_entries_batch,axis=0)
    return bed_entries,scores    

def compute_interpretation_scores(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    generators,out_suffixes=get_generators(args)
    print("created data generator(s)")
    model=get_model(args)
    print("loaded model") 
    
    for index in range(len(generators)):
        scores,bed_entries=interpret(generators[index],model,args)
        print("writing output file")
        np.savez_compressed(args.output_npz_file+'.'+out_suffixes[index],bed_entries=bed_entries,interp_scores=scores)


def main():
    args=parse_args()
    compute_interpretation_scores(args) 


if __name__=="__main__":
    main()
    
