#Raw keras model
import importlib
import imp
import argparse
import numpy as np
import h5py
from .generators import *
from .config import args_object_from_args_dict
import pdb

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--multi_gpu",action="store_true",default=False) 
    parser.add_argument("--data_path")
    parser.add_argument("--train_chroms",nargs="*",default=None)
    parser.add_argument("--validation_chroms",nargs="*",default=None) 
    parser.add_argument("--train_path")
    parser.add_argument("--valid_path")
    parser.add_argument("--model_hdf5")
    parser.add_argument("--batch_size",type=int,default=1000)
    parser.add_argument("--init_weights",default=None)
    parser.add_argument("--num_train",type=int,default=700000)
    parser.add_argument("--num_valid",type=int,default=150000)
    parser.add_argument("--ref_fasta",default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
    parser.add_argument('--w1',nargs="*", type=float, default=None)
    parser.add_argument('--w0',nargs="*", type=float, default=None)
    parser.add_argument("--w1_w0_file",default=None)
    parser.add_argument("--save_w1_w0", default=None,help="output text file to save w1 and w0 to")
    parser.add_argument("--weighted",action="store_true")
    parser.add_argument("--from_checkpoint_weights",default=None)
    parser.add_argument("--from_checkpoint_arch",default=None)
    parser.add_argument("--num_tasks",type=int)
    #add functionality to train on individuals' allele frequencies
    parser.add_argument("--vcf_file",default=None)
    parser.add_argument("--global_vcf",action="store_true")
    parser.add_argument("--revcomp",action="store_true")
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--patience",type=int,default=3)
    parser.add_argument("--patience_lr",type=int,default=2,help="number of epochs with no drop in validation loss after which to reduce lr")
    parser.add_argument("--architecture_spec",type=str,default="basset_architecture_multitask")
    parser.add_argument("--architecture_from_file",type=str,default=None)
    parser.add_argument("--tensorboard",action="store_true")
    parser.add_argument("--tensorboard_logdir",default="logs")
    parser.add_argument("--squeeze_input_for_gru",action="store_true")
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--train_upsample", type=float, default=None)
    parser.add_argument("--valid_upsample", type=float, default=None)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--max_queue_size",type=int,default=100)
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
    from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
    from keras.utils import multi_gpu_model
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
    np.random.seed(1234)
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


def train(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args)

    if args.train_path==None:
        args.train_path=args.data_path
    if args.valid_path==None:
        args.valid_path=args.data_path

    if args.train_upsample==None:
        train_upsample=False
        train_upsample_ratio=0
    else:
        train_upsample=True
        train_upsample_ratio=args.train_upsample 
    train_generator=DataGenerator(args.train_path,
                                  args.ref_fasta,
                                  upsample=train_upsample,
                                  upsample_ratio=train_upsample_ratio,
                                  chroms_to_use=args.train_chroms,
                                  get_w1_w0=args.weighted)
    print("generated training data generator!")
    if args.valid_upsample==None:
        valid_upsample=False
        valid_upsample_ratio=0
    else:
        valid_upsample=True
        valid_upsample_ratio=args.valid_upsample
    valid_generator=DataGenerator(args.valid_path,
                                   args.ref_fasta,
                                   upsample=valid_upsample,
                                   upsample_ratio=valid_upsample_ratio,
                                   chroms_to_use=args.validation_chroms)
    print("generated validation data generator!")
    w1,w0=get_weights(args,train_generator)
    
    try:
        if (args.architecture_from_file!=None):
            architecture_module=imp.load_source('',args.architecture_from_file)
        else:
            architecture_module=importlib.import_module('kerasAC.architectures.'+args.architecture_spec)
    except:
        raise Exception("could not import requested architecture, is it installed in kerasAC/kerasAC/architectures? Is the file with the requested architecture specified correctly?")
    model=architecture_module.getModelGivenModelOptionsAndWeightInits(w0,
                                                                      w1,
                                                                      args.init_weights,
                                                                      args.from_checkpoint_weights,
                                                                      args.from_checkpoint_arch,
                                                                      args.num_tasks,args.seed)
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
