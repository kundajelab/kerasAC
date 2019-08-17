import importlib
import imp
import argparse
import numpy as np
import h5py
from .generators import *
from .config import args_object_from_args_dict
from .tiledb_generators import * 
import pdb
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.utils import multi_gpu_model

def parse_args():
    parser=argparse.ArgumentParser()

    tiledbgroup=parser.add_argument_group('tiledb')
    tiledbgroup.add_argument("--chrom_sizes",default=None,help="chromsizes file for use with tiledb generator")
    tiledbgroup.add_argument("--label_source_attribute",default="fc_bigwig",help="tiledb attribute for use in label generation i.e. fc_bigwig")
    tiledbgroup.add_argument("--label_flank",type=int,help="flank around bin center to use in generating labels")
    tiledbgroup.add_argument("--label_aggregation",default=None,help="one of None, 'avg','max'")
    tiledbgroup.add_argument("--sequence_flank",type=int,help="length of sequence around bin center to use in one-hot-encoding")
    tiledbgroup.add_argument("--partition_attribute_for_upsample",default="idr_peak",help="tiledb attribute to use for upsampling, i.e. idr_peak")
    tiledbgroup.add_argument("--partition_thresh_for_upsample",type=float,default=1,help="values >= partition_thresh_for_upsample within the partition_attribute_for_upsample will be upsampled during training")

    input_data_path=parser.add_argument_group('input_data_path')
    input_data_path.add_argument("--tiledb_tasks_file",default=None,help="tsv file containing paths to tiledb tasks")
    input_data_path.add_argument("--data_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels")
    input_data_path.add_argument("--nonzero_bin_path",default=None,help="seqdataloader output file containing genome bins with non-zero values")
    input_data_path.add_argument("--universal_negative_path",default=None,help="seqdataloader output file containing genome bins that are universal negatives")
    input_data_path.add_argument("--train_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels for the training split")
    input_data_path.add_argument("--valid_path",default=None,help="seqdataloader output hdf5, or tsv file containing binned labels for the validation split")
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
    batch_params.add_argument("--label_transformer",default=None,help="transformation to apply to label values (i.e. log, asinh, etc). NOT IMPLEMENTED YET!")
    batch_params.add_argument("--squeeze_input_for_gru",action="store_true")
    batch_params.add_argument("--expand_dims",default=True)
    batch_params.add_argument("--upsample_thresh",type=float, default=0)
    batch_params.add_argument("--train_upsample", type=float, default=None)
    batch_params.add_argument("--valid_upsample", type=float, default=None)

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
    parallelization_params.add_argument("--multi_gpu",action="store_true",default=False) 

    vis_params=parser.add_argument_group("visualization")            
    vis_params.add_argument("--tensorboard",action="store_true")
    vis_params.add_argument("--tensorboard_logdir",default="logs")
    
    parser.add_argument("--model_hdf5",help="output model file that is generated at the end of training (in hdf5 format)")
    parser.add_argument("--seed",type=int,default=1234)    
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
    model_output_path = args.model_hdf5
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1,restore_best_weights=True)
    csvlogger = CSVLogger(args.model_hdf5+".log", append = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=args.patience_lr, min_lr=0.00000001)
    cur_callbacks=[checkpointer,earlystopper,csvlogger,reduce_lr]
    if args.tensorboard==True:
        from keras.callbacks import TensorBoard
        #create the specified logdir
        import os
        cur_logdir='/'.join([args.tensorboard_logdir,args.model_hdf5+'.tb'])
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
                        use_multiprocessing=True,
                        workers=args.threads,
                        max_queue_size=args.max_queue_size,
                        callbacks=cur_callbacks)
    model.save_weights(model_output_path+".weights")
    architecture_string=model.to_json()
    outf=open(args.model_hdf5+".arch",'w')
    outf.write(architecture_string)
    print("complete!!")

def initialize_generators_tiledb(args):
    train_generator=TiledbGenerator(chroms=args.train_chroms,
                                    chrom_sizes=args.chrom_sizes,
                                    ref_fasta=args.ref_fasta,
                                    shuffle_epoch_start=args.shuffle_epoch_start,
                                    shuffle_epoch_end=args.shuffle_epoch_end,
                                    batch_size=args.batch_size,
                                    task_file=args.tiledb_tasks_file,
                                    label_source=args.label_source_attribute,
                                    label_flank=args.label_flank,
                                    label_aggregation=args.label_aggregation,
                                    sequence_flank=args.sequence_flank,
                                    partition_attribute_for_upsample=args.partition_attribute_for_upsample,
                                    partition_thresh_for_upsample=args.partition_thresh_for_upsample,
                                    upsample_ratio=args.train_upsample,
                                    revcomp=args.revcomp,
                                    label_transformer=args.label_transformer)
    print("generated training data generator!")
    
    valid_generator=TiledbGenerator(chroms=args.validation_chroms,
                                    chrom_sizes=args.chrom_sizes,
                                    ref_fasta=args.ref_fasta,
                                    shuffle_epoch_start=args.shuffle_epoch_start,
                                    shuffle_epoch_end=args.shuffle_epoch_end,
                                    batch_size=args.batch_size,
                                    task_file=args.tiledb_tasks_file,
                                    label_source=args.label_source_attribute,
                                    label_flank=args.label_flank,
                                    label_aggregation=args.label_aggregation,
                                    sequence_flank=args.sequence_flank,
                                    partition_attribute_for_upsample=args.partition_attribute_for_upsample,
                                    partition_thresh_for_upsample=args.partition_thresh_for_upsample,
                                    upsample_ratio=args.valid_upsample,
                                    revcomp=args.revcomp,
                                    label_transformer=args.label_transformer)
    print("generated validation data generator")
    return train_generator, valid_generator

def get_paths(args):
    if args.train_path is None:
        train_path=args.data_path
    else:
        train_path=args.train_path
        
    if args.valid_path is None:
        valid_path=args.data_path
    else:
        valid_path=args.valid_path
        
    return train_path, valid_path

def initialize_generators(args):
    #get upsampling parameters
    train_path, valid_path=get_paths(args)
    
    #data is being read in from tiledb for training 
    if args.tiledb_tasks_file is not None:
        return initialize_generators_tiledb(args)
    
    train_generator=DataGenerator(data_path=train_path,
                                  nonzero_bin_path=args.nonzero_bin_path,
                                  universal_negative_path=args.universal_negative_path,
                                  ref_fasta=args.ref_fasta,
                                  batch_size=args.batch_size,
                                  add_revcomp=args.revcomp,
                                  upsample_thresh=args.upsample_thresh,
                                  upsample_ratio=args.train_upsample,
                                  chroms_to_use=args.train_chroms,
                                  get_w1_w0=args.weighted,
                                  expand_dims=args.expand_dims,
                                  tasks=args.tasks)
    
    print("generated training data generator!")
    valid_generator=DataGenerator(data_path=valid_path,
                                  nonzero_bin_path=args.nonzero_bin_path,
                                  universal_negative_path=args.universal_negative_path,
                                  ref_fasta=args.ref_fasta,
                                  batch_size=args.batch_size,
                                  add_revcomp=args.revcomp,                        
                                  upsample_thresh=args.upsample_thresh,
                                  upsample_ratio=args.valid_upsample,
                                  chroms_to_use=args.validation_chroms,
                                  expand_dims=args.expand_dims,
                                  tasks=args.tasks)
    print("generated validation data generator!")
    return train_generator, valid_generator 
    
def train(args):
    
    if type(args)==type({}):
        args=args_object_from_args_dict(args)

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
    model=architecture_module.getModelGivenModelOptionsAndWeightInits(args)
    if args.multi_gpu==True:
        try:
            model=multi_gpu_model(model)
            print("Training on all available GPU's. Set args.multi_gpu = False to avoid this") 
        except:
            pass 
    print("compiled the model!")
    fit_and_evaluate(model,train_generator,
                     valid_generator,args)

def main():
    args=parse_args()
    train(args)
    

if __name__=="__main__":
    main()
