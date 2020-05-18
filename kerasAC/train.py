from __future__ import division, print_function, absolute_import
import importlib
import imp
import os
import tempfile
import argparse
import numpy as np
import h5py
from .s3_sync import *
from .generators.basic_generator import *
from .generators.tiledb_generator import *
from .custom_callbacks import * 
from .tiledb_config import *
from .get_model import *
from .splits import * 
from . import config
import pdb
from keras.callbacks import *
from keras.utils import multi_gpu_model
import gc
import multiprocessing
#multiprocessing.set_start_method('forkserver', force=True)
def parse_args():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--model_prefix",help="output model file that is generated at the end of training (in hdf5 format)")
    parser.add_argument("--seed",type=int,default=1234)    
    parser.add_argument("--num_inputs",type=int)
    parser.add_argument("--num_outputs",type=int)
    parser.add_argument("--use_multiprocessing",type=bool,default=True)
    parser.add_argument("--chrom_sizes",default=None,help="chromsizes file for use with tiledb generator")
    
    tiledbgroup=parser.add_argument_group('tiledb')
    tiledbgroup.add_argument("--tdb_array",help="name of tdb array to use")

    tiledbgroup.add_argument("--tdb_output_source_attribute",nargs="+",help="tiledb attribute for use in label generation i.e. fc_bigwig")
    tiledbgroup.add_argument("--tdb_output_min",nargs="*", default=None)
    tiledbgroup.add_argument("--tdb_output_max",nargs="*", default=None)        
    tiledbgroup.add_argument("--tdb_output_flank",nargs="+",type=int,help="flank around bin center to use in generating outputs")
    tiledbgroup.add_argument("--tdb_output_aggregation",nargs="+",help="method for output aggregation; one of None, 'avg','max'")
    tiledbgroup.add_argument("--tdb_output_transformation",nargs="+",help="method for output transformation; one of None, 'log','log10','asinh'")
    tiledbgroup.add_argument("--tdb_transformation_pseudocount",type=float,default=0.001)
    
    tiledbgroup.add_argument("--tdb_input_source_attribute",nargs="+",help="attribute to use for generating model input, or 'seq' for one-hot-encoded sequence")
    tiledbgroup.add_argument("--tdb_input_min",nargs="*", default=None)
    tiledbgroup.add_argument("--tdb_input_max",nargs="*", default=None)    
    tiledbgroup.add_argument("--tdb_input_flank",nargs="+",type=int,help="length of sequence around bin center to use for input")
    tiledbgroup.add_argument("--tdb_input_aggregation",nargs="+",help="method for input aggregation; one of 'None','avg','max'")
    tiledbgroup.add_argument("--tdb_input_transformation",nargs="+",help="method for input transformation; one of None, 'log','log10','asinh'")

    tiledbgroup.add_argument("--tdb_partition_attribute_for_upsample",default=None,help="tiledb attribute to use for upsampling, i.e. idr_peak")
    tiledbgroup.add_argument("--tdb_partition_thresh_for_upsample",type=float,default=None,help="values >= partition_thresh_for_upsample within the partition_attribute_for_upsample will be upsampled during training")

    tiledbgroup.add_argument("--tdb_ambig_attribute",default=None,help="attribute indicating ambiguous regions to not train on")    

    tiledbgroup.add_argument("--tdb_bias_arrays",nargs="*",default=None)
    tiledbgroup.add_argument("--tdb_bias_source_attribute",nargs="*")
    tiledbgroup.add_argument("--tdb_bias_flank",nargs="*",type=int)
    tiledbgroup.add_argument("--tdb_bias_aggregation",nargs="*")
    tiledbgroup.add_argument("--tdb_bias_transformation",nargs="*")
    tiledbgroup.add_argument("--tdb_bias_pseudocount",type=float,default=0.001)
    
    input_data_path=parser.add_argument_group('input_data_path')
    input_data_path.add_argument("--index_data_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels")
    input_data_path.add_argument("--index_train_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels for the training split")
    input_data_path.add_argument("--index_valid_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels for the validation split")
    input_data_path.add_argument("--index_tasks",nargs="*",default=None)
    
    input_data_path.add_argument("--input_data_path",nargs="+",default=None,help="seq or path to seqdataloader hdf5")
    input_data_path.add_argument("--input_train_path",nargs="+",default=None,help="seq or seqdataloader hdf5")
    input_data_path.add_argument("--input_valid_path",nargs="+",default=None,help="seq or seqdataloader hdf5")

    input_data_path.add_argument("--output_data_path",nargs="+",default=None,help="path to seqdataloader hdf5")
    input_data_path.add_argument("--output_train_path",nargs="+",default=None,help="seqdataloader hdf5")
    input_data_path.add_argument("--output_valid_path",nargs="+",default=None,help="seqdataloader hdf5")
    
    input_data_path.add_argument("--ref_fasta",default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
    
    train_val_splits=parser.add_argument_group("train_val_splits")
    train_val_splits.add_argument("--train_chroms",nargs="*",default=None)
    train_val_splits.add_argument("--validation_chroms",nargs="*",default=None)
    train_val_splits.add_argument("--genome",default=None)
    train_val_splits.add_argument("--fold",type=int,default=None)
    train_val_splits.add_argument("--bed_regions_train",default=None)
    train_val_splits.add_argument("--bed_regions_validate",default=None)
    train_val_splits.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    train_val_splits.add_argument("--num_train",type=int,default=700000)
    train_val_splits.add_argument("--num_valid",type=int,default=150000)

    weights_params=parser.add_argument_group("weights_params")
    weights_params.add_argument("--load_model_hdf5")
    weights_params.add_argument("--weights",default=None)
    weights_params.add_argument("--init_weights",default=None,help="legacy, will be deprecated in next release") 
    weights_params.add_argument('--w1',nargs="*", type=float, default=None)
    weights_params.add_argument('--w0',nargs="*", type=float, default=None)
    weights_params.add_argument("--w1_w0_file",default=None)
    weights_params.add_argument("--save_w1_w0", default=None,help="output text file to save w1 and w0 to")
    weights_params.add_argument("--weighted",action="store_true",help="separate task-specific weights denoted with w1, w0 args are to be used")
    
    
    arch_params=parser.add_argument_group("arch_params")
    arch_params.add_argument("--json",default=None)
    arch_params.add_argument("--yaml",default=None)
    arch_params.add_argument("--architecture_spec",type=str,default="basset_architecture_multitask")
    arch_params.add_argument("--architecture_from_file",type=str,default=None)
    arch_params.add_argument("--model_params",type=str,default=None,help="2-column file with param name in column 1 and param value in column 2")
    arch_params.add_argument("--num_tasks",type=int)
    arch_params.add_argument("--tasks",nargs="*",default=None,help="list of tasks to train on, by name")
    arch_params.add_argument("--task_indices",nargs="*",default=None,help="list of tasks to train on, by index of their position in tdb matrix")
    
    batch_params=parser.add_argument_group("batch_params")
    batch_params.add_argument("--batch_size",type=int,default=1000)
    batch_params.add_argument("--revcomp",action="store_true")
    batch_params.add_argument("--label_transformer",nargs="+",default=None,help="transformation to apply to label values")
    batch_params.add_argument("--squeeze_input_for_gru",action="store_true")
    batch_params.add_argument("--expand_dims",default=False,action="store_true")
    batch_params.add_argument("--upsample_thresh_list_train",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_ratio_list_train",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_thresh_list_eval",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_ratio_list_eval",type=float,nargs="*",default=None)
    batch_params.add_argument("--upsample_threads",type=int,default=1)
    
    epoch_params=parser.add_argument_group("epoch_params")
    epoch_params.add_argument("--epochs",type=int,default=40)
    epoch_params.add_argument("--patience",type=int,default=3)
    epoch_params.add_argument("--patience_lr",type=int,default=2,help="number of epochs with no drop in validation loss after which to reduce lr")
    epoch_params.add_argument("--shuffle_epoch_start",type=bool, default=True)
    epoch_params.add_argument("--shuffle_epoch_end",type=bool, default=False)
    
    #add functionality to train on individuals' allele frequencies
    snp_params=parser.add_argument_group("snp_params")
    snp_params.add_argument("--vcf_file",default=None)
    snp_params.add_argument("--global_vcf",action="store_true")

    parallelization_params=parser.add_argument_group("parallelization")
    parallelization_params.add_argument("--threads",type=int,default=1)
    parallelization_params.add_argument("--max_queue_size",type=int,default=100)
    parallelization_params.add_argument("--num_gpus",type=int,default=1)

    vis_params=parser.add_argument_group("visualization")            
    vis_params.add_argument("--tensorboard",action="store_true")
    vis_params.add_argument("--tensorboard_logdir",default="logs")
    vis_params.add_argument("--trackables",nargs="*",default=['loss','val_loss'], help="list of things to track per batch, such as logcount_predictions_loss,loss,profile_predictions_loss,val_logcount_predictions_loss,val_loss,val_profile_predictions_loss")
    return parser.parse_args()


def fit_and_evaluate(model,train_gen,valid_gen,args):
    #accomodate storage on s3
    if args.model_prefix.startswith('s3'):
        #store in local temporary file
        model_output_path_string=os.path.basename(args.model_prefix)
        model_output_path_hdf5=tempfile.NamedTemporaryFile(suffix=model_output_path_string+".hdf5")
        model_output_path_logs=tempfile.NamedTemporaryFile(suffix=model_output_path_string+".log")        
        model_output_path_arch=tempfile.NamedTemporaryFile(suffix=model_output_path_string+".arch")
        model_output_path_weights=tempfile.NamedTemporaryFile(suffix=model_output_path_string+".weights")
        model_output_path_hdf5_name=model_output_path_hdf5.name
        model_output_path_logs_name=model_output_path_logs.name
        model_output_path_arch_name=model_output_path_arch.name
        model_output_path_weights_name=model_output_path_weights.name
        
    else: 
        model_output_path_string = args.model_prefix
        model_output_path_hdf5_name=model_output_path_string+".hdf5"
        model_output_path_logs_name=model_output_path_string+".log"
        model_output_path_arch_name=model_output_path_string+".arch"
        model_output_path_weights_name=model_output_path_string+".weights"

    
    checkpointer = ModelCheckpoint(filepath=model_output_path_hdf5_name, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1,restore_best_weights=True)
    history=LossHistory(model_output_path_logs_name+".batch",args.trackables)
    csvlogger = CSVLogger(model_output_path_logs_name, append = False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=args.patience_lr, min_lr=0.00000001)
    cur_callbacks=[checkpointer,earlystopper,csvlogger,reduce_lr,history]
    if args.tensorboard==True:
        from keras.callbacks import TensorBoard
        cur_logdir='/'.join([args.tensorboard_logdir,model_output_path_string.split('/')[-1]+'.tb'])
        if not os.path.exists(cur_logdir):
                os.makedirs(cur_logdir)
        tensorboard_visualizer=TensorBoard(log_dir=cur_logdir, histogram_freq=0, batch_size=500, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        cur_callbacks.append(tensorboard_visualizer)
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        steps_per_epoch=args.num_train/args.batch_size,
                        validation_steps=args.num_valid/args.batch_size,
                        epochs=args.epochs,
                        verbose=1,
                        use_multiprocessing=args.use_multiprocessing,
                        workers=args.threads,
                        max_queue_size=args.max_queue_size,
                        callbacks=cur_callbacks,
                        shuffle=False)
    print('fit_generator complete') 
    model.save_weights(model_output_path_weights_name)
    print('weights saved') 
    architecture_string=model.to_json()
    with open(model_output_path_arch_name,'w') as outf:
        outf.write(architecture_string)
    print('saved model architecture') 
    #sync to s3 if needed
    if args.model_prefix.startswith('s3://'):
        #sync log, model hdf5, weight file, arch file
        upload_s3_file(args.model_prefix+'.hdf5',model_output_path_hdf5_name)
        upload_s3_file(args.model_prefix+'.arch',model_output_path_arch_name)
        upload_s3_file(args.model_prefix+'.log',model_output_path_logs_name)
        upload_s3_file(args.model_prefix+'.weights',model_output_path_weights_name)                
    print("complete!!")
    
def initializer_generators_hdf5(args):
    #get upsampling parameters
    train_chroms=get_chroms(args,split='train')
    index_train_path, index_valid_path, input_train_path, input_valid_path, output_train_path, output_valid_path=get_paths(args)
    train_generator=DataGenerator(index_path=index_train_path,
                                  input_path=input_train_path,
                                  output_path=output_train_path,
                                  index_tasks=args.index_tasks,
                                  num_inputs=args.num_inputs,
                                  num_outputs=args.num_outputs,
                                  ref_fasta=args.ref_fasta,
                                  batch_size=args.batch_size,
                                  add_revcomp=args.revcomp,
                                  chroms_to_use=train_chroms,
                                  get_w1_w0=args.weighted,
                                  expand_dims=args.expand_dims,
                                  upsample_thresh_list=args.upsample_thresh_list_train,
                                  upsample_ratio_list=args.upsample_ratio_list_train,
                                  tasks=args.tasks)

    
    print("generated training data generator!")
    valid_chroms=get_chroms(args,split='valid')
    valid_generator=DataGenerator(index_path=index_train_path,
                                  input_path=input_train_path,
                                  output_path=output_train_path,
                                  index_tasks=args.index_tasks,
                                  num_inputs=args.num_inputs,
                                  num_outputs=args.num_outputs,
                                  ref_fasta=args.ref_fasta,
                                  batch_size=args.batch_size,
                                  add_revcomp=args.revcomp,                        
                                  upsample_thresh_list=args.upsample_thresh_list_eval,
                                  upsample_ratio_list=args.upsample_ratio_list_eval,
                                  chroms_to_use=valid_chroms,
                                  expand_dims=args.expand_dims,
                                  tasks=args.tasks)
    print("generated validation data generator!")
    return train_generator, valid_generator 

def initialize_generators_tiledb(args):
    #open array for reading
    #print("consolidating:")
    #import tiledb
    #tiledb.consolidate(args.tdb_array)
    #print("done")

    if args.upsample_ratio_list_train is not None:
        upsample_ratio_train=args.upsample_ratio_list_train[0]
        print("warning! only a single ratio for upsampling supported for tiledb as of now")
    else:
        upsample_ratio_train=None
    if args.upsample_ratio_list_eval is not None:
        upsample_ratio_eval=args.upsample_ratio_list_eval[0]
        print("warning! only a single ratio for upsampling supported for tiledb as of now")
    else:
        upsample_ratio_eval=None
    import tiledb
    tdb_config=get_default_config() 
    tdb_ctx=tiledb.Ctx(config=tdb_config)
    train_chroms=get_chroms(args,split='train')
    train_generator=TiledbGenerator(chroms=train_chroms,
                                    ref_fasta=args.ref_fasta,
                                    shuffle_epoch_start=args.shuffle_epoch_start,
                                    shuffle_epoch_end=args.shuffle_epoch_end,
                                    batch_size=args.batch_size,
                                    tdb_array=args.tdb_array,
                                    tdb_partition_attribute_for_upsample=args.tdb_partition_attribute_for_upsample,
                                    tdb_partition_thresh_for_upsample=args.tdb_partition_thresh_for_upsample,
                                    tdb_input_source_attribute=args.tdb_input_source_attribute,
                                    tdb_input_flank=args.tdb_input_flank,
                                    tdb_input_min=args.tdb_input_min,
                                    tdb_input_max=args.tdb_input_max,
                                    tdb_input_aggregation=args.tdb_input_aggregation,
                                    tdb_input_transformation=args.tdb_input_transformation,
                                    pseudocount=args.tdb_transformation_pseudocount,
                                    tdb_output_source_attribute=args.tdb_output_source_attribute,
                                    tdb_output_flank=args.tdb_output_flank,
                                    tdb_output_min=args.tdb_output_min,
                                    tdb_output_max=args.tdb_output_max,
                                    tdb_output_aggregation=args.tdb_output_aggregation,
                                    tdb_output_transformation=args.tdb_output_transformation,
                                    tdb_ambig_attribute=args.tdb_ambig_attribute,
                                    tdb_bias_arrays=args.tdb_bias_arrays,
                                    tdb_bias_source_attribute=args.tdb_bias_source_attribute,
                                    tdb_bias_flank=args.tdb_bias_flank,
                                    tdb_bias_aggregation=args.tdb_bias_aggregation,
                                    tdb_bias_transformation=args.tdb_bias_transformation,
                                    bias_pseudocount=args.tdb_bias_pseudocount,
                                    tasks=args.tasks,
                                    task_indices=args.task_indices,
                                    upsample_ratio=upsample_ratio_train,
                                    num_inputs=args.num_inputs,
                                    num_outputs=args.num_outputs,
                                    expand_dims=args.expand_dims,
                                    add_revcomp=args.revcomp,
                                    tdb_config=tdb_config,
                                    tdb_ctx=tdb_ctx,
                                    num_threads=args.upsample_threads,
                                    bed_regions=args.bed_regions_train,
                                    bed_regions_summit_center=args.center_on_summit)
    
    print("generated training data generator!")
    valid_chroms=get_chroms(args,split='valid')
    valid_generator=TiledbGenerator(chroms=valid_chroms,
                                    ref_fasta=args.ref_fasta,
                                    shuffle_epoch_start=args.shuffle_epoch_start,
                                    shuffle_epoch_end=args.shuffle_epoch_end,
                                    batch_size=args.batch_size,
                                    tdb_array=args.tdb_array,
                                    tdb_partition_attribute_for_upsample=args.tdb_partition_attribute_for_upsample,
                                    tdb_partition_thresh_for_upsample=args.tdb_partition_thresh_for_upsample,
                                    tdb_input_source_attribute=args.tdb_input_source_attribute,
                                    tdb_input_flank=args.tdb_input_flank,
                                    tdb_input_min=args.tdb_input_min,
                                    tdb_input_max=args.tdb_input_max,
                                    tdb_input_aggregation=args.tdb_input_aggregation,
                                    tdb_input_transformation=args.tdb_input_transformation,
                                    pseudocount=args.tdb_transformation_pseudocount,
                                    tdb_output_source_attribute=args.tdb_output_source_attribute,
                                    tdb_output_flank=args.tdb_output_flank,
                                    tdb_output_min=args.tdb_output_min,
                                    tdb_output_max=args.tdb_output_max,
                                    tdb_output_aggregation=args.tdb_output_aggregation,
                                    tdb_output_transformation=args.tdb_output_transformation,
                                    tdb_ambig_attribute=args.tdb_ambig_attribute,
                                    tdb_bias_arrays=args.tdb_bias_arrays,
                                    tdb_bias_source_attribute=args.tdb_bias_source_attribute,
                                    tdb_bias_flank=args.tdb_bias_flank,
                                    tdb_bias_aggregation=args.tdb_bias_aggregation,
                                    tdb_bias_transformation=args.tdb_bias_transformation,                                    
                                    tasks=args.tasks,
                                    task_indices=args.task_indices,
                                    upsample_ratio=upsample_ratio_eval,
                                    num_inputs=args.num_inputs,
                                    num_outputs=args.num_outputs,
                                    expand_dims=args.expand_dims,
                                    add_revcomp=args.revcomp,
                                    tdb_config=tdb_config,
                                    tdb_ctx=tdb_ctx,
                                    num_threads=args.upsample_threads,
                                    bed_regions=args.bed_regions_validate,
                                    bed_regions_summit_center=args.center_on_summit)
    
    print("generated validation data generator")
    return train_generator, valid_generator

def get_paths(args):
    if args.index_train_path is None:
        index_train_path=args.index_data_path
    else:
        index_train_path=args.index_train_path        
    if args.input_train_path is None:
        input_train_path=args.input_data_path
    else:
        input_train_path=args.input_train_path        
    if args.output_train_path is None:
        output_train_path=args.output_data_path
    else:
        output_train_path=args.output_train_path        
    if args.index_valid_path is None:
        index_valid_path=args.index_data_path
    else:
        index_valid_path=args.index_valid_path
    if args.input_valid_path is None:
        input_valid_path=args.input_data_path
    else:
        input_valid_path=args.input_valid_path
    if args.output_valid_path is None:
        output_valid_path=args.output_data_path
    else:
        output_valid_path=args.output_valid_path        
    return index_train_path, index_valid_path, input_train_path, input_valid_path, output_train_path, output_valid_path 

def initialize_generators(args):    
    #data is being read in from tiledb for training
    print(args)
    if args.tdb_array is not None:
        return initialize_generators_tiledb(args)
    else:
        return initializer_generators_hdf5(args)
    
    
def train(args):
    if type(args)==type({}):
        args=config.args_object_from_args_dict(args)


    #create the generators
    train_generator,valid_generator=initialize_generators(args)
    
    w1,w0=get_w1_w0_training(args,train_generator)
    args.w1=w1
    args.w0=w0
    model=get_model(args)
    fit_and_evaluate(model,train_generator,
                     valid_generator,args)

    #remove any temporary s3 files
    print("running cleanup!") 
    run_cleanup()
    
def main():
    gc.freeze()
    args=parse_args()
    train(args)
    print("Exiting!")

if __name__=="__main__":
    main()
