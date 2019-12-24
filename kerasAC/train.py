from __future__ import division, print_function, absolute_import
import importlib
import imp
import os
import tempfile
import argparse
import numpy as np
import h5py
import boto3
from .generators.basic_generator import *
from .generators.tiledb_generator import *
from . import config
import pdb
from keras.callbacks import *
from keras.utils import multi_gpu_model
import gc
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)
def parse_args():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--model_prefix",help="output model file that is generated at the end of training (in hdf5 format)")
    parser.add_argument("--seed",type=int,default=1234)    
    parser.add_argument("--num_inputs",type=int)
    parser.add_argument("--num_outputs",type=int)
    parser.add_argument("--use_multiprocessing",type=bool,default=True)
    
    tiledbgroup=parser.add_argument_group('tiledb')
    tiledbgroup.add_argument("--chrom_sizes",default=None,help="chromsizes file for use with tiledb generator")    
    tiledbgroup.add_argument("--tdb_outputs",nargs="+")
    tiledbgroup.add_argument("--tdb_output_source_attribute",nargs="+",default="fc_bigwig",help="tiledb attribute for use in label generation i.e. fc_bigwig")
    tiledbgroup.add_argument("--tdb_output_flank",nargs="+",type=int,help="flank around bin center to use in generating outputs")
    tiledbgroup.add_argument("--tdb_output_aggregation",nargs="+",help="method for output aggregation; one of None, 'avg','max'")
    tiledbgroup.add_argument("--tdb_output_transformation",nargs="+",help="method for output transformation; one of None, 'log','log10','asinh'")
    
    tiledbgroup.add_argument("--tdb_inputs",nargs="+")
    tiledbgroup.add_argument("--tdb_input_source_attribute",nargs="+",help="attribute to use for generating model input, or 'seq' for one-hot-encoded sequence")
    tiledbgroup.add_argument("--tdb_input_flank",nargs="+",type=int,help="length of sequence around bin center to use for input")
    tiledbgroup.add_argument("--tdb_input_aggregation",nargs="+",help="method for input aggregation; one of 'None','avg','max'")
    tiledbgroup.add_argument("--tdb_input_transformation",nargs="+",help="method for input transformation; one of None, 'log','log10','asinh'")

    tiledbgroup.add_argument("--tdb_indexer",default=None,help="tiledb paths for each input task")
    tiledbgroup.add_argument("--tdb_partition_attribute_for_upsample",default="idr_peak",help="tiledb attribute to use for upsampling, i.e. idr_peak")
    tiledbgroup.add_argument("--tdb_partition_thresh_for_upsample",type=float,default=1,help="values >= partition_thresh_for_upsample within the partition_attribute_for_upsample will be upsampled during training")
        
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
    train_val_splits.add_argument("--num_train",type=int,default=700000)
    train_val_splits.add_argument("--num_valid",type=int,default=150000)

    weights_params=parser.add_argument_group("weights_params")
    weights_params.add_argument("--init_weights",default=None)
    weights_params.add_argument('--w1',nargs="*", type=float, default=None)
    weights_params.add_argument('--w0',nargs="*", type=float, default=None)
    weights_params.add_argument("--w1_w0_file",default=None)
    weights_params.add_argument("--save_w1_w0", default=None,help="output text file to save w1 and w0 to")
    weights_params.add_argument("--weighted",action="store_true",help="separate task-specific weights denoted with w1, w0 args are to be used")
    weights_params.add_argument("--from_checkpoint_weights",default=None)
    
    arch_params=parser.add_argument_group("arch_params")
    arch_params.add_argument("--from_checkpoint_arch",default=None)
    arch_params.add_argument("--architecture_spec",type=str,default="basset_architecture_multitask")
    arch_params.add_argument("--architecture_from_file",type=str,default=None)
    arch_params.add_argument("--num_tasks",type=int)
    arch_params.add_argument("--tasks",nargs="*",default=None)
    
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

    epoch_params=parser.add_argument_group("epoch_params")
    epoch_params.add_argument("--epochs",type=int,default=40)
    epoch_params.add_argument("--patience",type=int,default=3)
    epoch_params.add_argument("--patience_lr",type=int,default=2,help="number of epochs with no drop in validation loss after which to reduce lr")
    epoch_params.add_argument("--shuffle_epoch_start",type=bool, default=True)
    epoch_params.add_argument("--shuffle_epoch_end",type=bool, default=True)
    
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
    return parser.parse_args()

def get_weights(args,train_generator):
    w1=args.w1
    w0=args.w0
    w1_w0_file=args.w1_w0_file
    if (args.weighted==True and (w1==None or w0==None) ):
        if args.w1_w0_file==None:
            w1=train_generator.w1
            w0=train_generator.w0        
            assert args.save_w1_w0 !=None
            with open(args.save_w1_w0, 'w') as weight_file:
                for i in range(len(w1)):
                    weight_file.write(str(w1[i])+'\t'+str(w0[i])+'\n')
        else:
            w1_w0=np.loadtxt(args.w1_w0_file)
            w1=list(w1_w0[:,0])
            w0=list(w1_w0[:,1]) 
        print("got weights!")
    return w1,w0

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
    csvlogger = CSVLogger(model_output_path_logs_name, append = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=args.patience_lr, min_lr=0.00000001)
    cur_callbacks=[checkpointer,earlystopper,csvlogger,reduce_lr]
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
    model.save_weights(model_output_path_weights_name)
    architecture_string=model.to_json()
    with open(model_output_path_arch_name,'w') as outf:
        outf.write(architecture_string)
    #sync to s3 if needed
    if args.model_prefix.startswith('s3://'):
        #sync log, model hdf5, weight file, arch file
        s3_client=boto3.client(service_name='s3')
        bucket=args.model_prefix.strip('s3://').split('/')[0]
        s3_prefix="/".join(args.model_prefix.strip("s3://").split("/")[1::])
        s3_hdf5=s3_prefix+'.hdf5'
        s3_arch=s3_prefix+'.arch'
        s3_log=s3_prefix+'.log'
        s3_weights=s3_prefix+'.weights'
        s3_client.upload_file(model_output_path_hdf5,bucket,s3_hdf5)
        s3_client.upload_file(model_output_path_arch,bucket,s3_arch)
        s3_client.upload_file(model_output_path_weights,bucket,s3_weights)
        s3_client.upload_file(model_output_path_logs,bucket,s3_log)  
    print("complete!!")
    
def initializer_generators_hdf5(args):
    #get upsampling parameters
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
                                  chroms_to_use=args.train_chroms,
                                  get_w1_w0=args.weighted,
                                  expand_dims=args.expand_dims,
                                  upsample_thresh_list=args.upsample_thresh_list_train,
                                  upsample_ratio_list=args.upsample_ratio_list_train,
                                  tasks=args.tasks)
    
    print("generated training data generator!")
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
                                  chroms_to_use=args.validation_chroms,
                                  expand_dims=args.expand_dims,
                                  tasks=args.tasks)
    print("generated validation data generator!")
    return train_generator, valid_generator 

def initialize_generators_tiledb(args):
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
    tdb_config=tiledb.Config()
    tdb_config['vfs.s3.region']='us-west-1'
    tdb_config["sm.check_coord_dups"]="false"
    tdb_config["sm.check_coord_oob"]="false"
    tdb_config["sm.check_global_order"]="false"
    tdb_config["sm.num_reader_threads"]="50"
    tdb_config["sm.num_async_threads"]="50"
    tdb_config["vfs.num_threads"]="50"    
    tdb_ctx=tiledb.Ctx(config=tdb_config)
    train_generator=TiledbGenerator(chroms=args.train_chroms,
                                    chrom_sizes=args.chrom_sizes,
                                    ref_fasta=args.ref_fasta,
                                    shuffle_epoch_start=args.shuffle_epoch_start,
                                    shuffle_epoch_end=args.shuffle_epoch_end,
                                    batch_size=args.batch_size,
                                    tdb_indexer=args.tdb_indexer,
                                    tdb_partition_attribute_for_upsample=args.tdb_partition_attribute_for_upsample,
                                    tdb_partition_thresh_for_upsample=args.tdb_partition_thresh_for_upsample,
                                    tdb_inputs=args.tdb_inputs,
                                    tdb_input_source_attribute=args.tdb_input_source_attribute,
                                    tdb_input_flank=args.tdb_input_flank,
                                    tdb_input_aggregation=args.tdb_input_aggregation,
                                    tdb_input_transformation=args.tdb_input_transformation,
                                    tdb_outputs=args.tdb_outputs,
                                    tdb_output_source_attribute=args.tdb_output_source_attribute,
                                    tdb_output_flank=args.tdb_output_flank,
                                    tdb_output_aggregation=args.tdb_output_aggregation,
                                    tdb_output_transformation=args.tdb_output_transformation,
                                    upsample_ratio=upsample_ratio_train,
                                    num_inputs=args.num_inputs,
                                    num_outputs=args.num_outputs,
                                    expand_dims=args.expand_dims,
                                    add_revcomp=args.revcomp,
                                    tdb_config=tdb_config,
                                    tdb_ctx=tdb_ctx)
    
    print("generated training data generator!")
    valid_generator=TiledbGenerator(chroms=args.validation_chroms,
                                    chrom_sizes=args.chrom_sizes,
                                    ref_fasta=args.ref_fasta,
                                    shuffle_epoch_start=args.shuffle_epoch_start,
                                    shuffle_epoch_end=args.shuffle_epoch_end,
                                    batch_size=args.batch_size,
                                    tdb_indexer=args.tdb_indexer,
                                    tdb_partition_attribute_for_upsample=args.tdb_partition_attribute_for_upsample,
                                    tdb_partition_thresh_for_upsample=args.tdb_partition_thresh_for_upsample,
                                    tdb_inputs=args.tdb_inputs,
                                    tdb_input_source_attribute=args.tdb_input_source_attribute,
                                    tdb_input_flank=args.tdb_input_flank,
                                    tdb_input_aggregation=args.tdb_input_aggregation,
                                    tdb_input_transformation=args.tdb_input_transformation,
                                    tdb_outputs=args.tdb_outputs,
                                    tdb_output_source_attribute=args.tdb_output_source_attribute,
                                    tdb_output_flank=args.tdb_output_flank,
                                    tdb_output_aggregation=args.tdb_output_aggregation,
                                    tdb_output_transformation=args.tdb_output_transformation,
                                    upsample_ratio=upsample_ratio_eval,
                                    num_inputs=args.num_inputs,
                                    num_outputs=args.num_outputs,
                                    expand_dims=args.expand_dims,
                                    add_revcomp=args.revcomp,
                                    tdb_config=tdb_config,
                                    tdb_ctx=tdb_ctx)
    
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
    if args.tdb_indexer is not None:
        return initialize_generators_tiledb(args)
    else:
        return initializer_generators_hdf5(args)
    
    
def train(args):
    if type(args)==type({}):
        args=config.args_object_from_args_dict(args)


    #create the generators
    train_generator,valid_generator=initialize_generators(args)
    w1,w0=get_weights(args,train_generator)
    args.w1=w1
    args.w0=w0
    try:
        if (args.architecture_from_file!=None):
            architecture_module=imp.load_source('',args.architecture_from_file)
        else:
            architecture_module=importlib.import_module('kerasAC.architectures.'+args.architecture_spec)
    except:
        raise Exception("could not import requested architecture, is it installed in kerasAC/kerasAC/architectures? Is the file with the requested architecture specified correctly?")
    model,optimizer,loss=architecture_module.getModelGivenModelOptionsAndWeightInits(args)
    if args.num_gpus >1:
       try:
           model=multi_gpu_model(model,gpus=args.num_gpus)
           #recompile
           model.compile(optimizer=optimizer,loss=loss) 
           print("Training on " +str(args.num_gpus)+" GPU's. Set args.multi_gpu = False to avoid this") 
       except:
           print("failed to instantiate multi-gpu model, defaulting to single-gpu model")
    print("compiled the model!")

    
    fit_and_evaluate(model,train_generator,
                     valid_generator,args)

def main():
    gc.freeze()
    args=parse_args()
    train(args)
    

if __name__=="__main__":
    main()
