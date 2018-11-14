#Raw keras model
import importlib
import imp
import argparse
import numpy as np
import h5py
from kerasAC.create_generators import *

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--valid_path")
    parser.add_argument("--model_output_file")
    parser.add_argument("--batch_size",type=int,default=1000)
    parser.add_argument("--init_weights",default=None)
    parser.add_argument("--num_train",type=int,default=700000)
    parser.add_argument("--num_valid",type=int,default=150000)
    parser.add_argument("--ref_fasta",default="/srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa")
    parser.add_argument("--w0_file",default=None)
    parser.add_argument("--w1_file",default=None)
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
    parser.add_argument("--upsample_ratio", type=float, default=0.5)
    parser.add_argument("--save_weights")
    return parser.parse_args()

def fit_and_evaluate(model,train_gen,valid_gen,args):
    model_output_path = args.model_output_file
    from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1)
    csvlogger = CSVLogger(args.model_output_file+".log", append = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=args.patience_lr, min_lr=0.00000001)
    cur_callbacks=[checkpointer,earlystopper,csvlogger,reduce_lr]
    if args.tensorboard==True:
        from keras.callbacks import TensorBoard
        #create the specified logdir
        import os
        cur_logdir='/'.join([args.tensorboard_logdir,args.model_output_file+'.tb'])
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
                        callbacks=cur_callbacks)
    model.save_weights(model_output_path+".weights")
    architecture_string=model.to_json()
    outf=open(args.model_output_file+".arch",'w')
    outf.write(architecture_string)
    print("complete!!")

def get_weights(bed_path, weights_path):
    import pandas as pd
    data=pd.read_csv(bed_path,header=0,sep='\t',index_col=[0,1,2])
    w1=[float(data.shape[0])/sum(data.iloc[:,i]==1) for i in range(data.shape[1])]
    w0=[float(data.shape[0])/sum(data.iloc[:,i]==0) for i in range(data.shape[1])]
    with open(weights_path, 'w') as weight_file:
        weight_file.write("--w1")
        for i in w1:
            weight_file.write(" " + str(i))
        weight_file.write(" \\" + "\n--w0")
        for i in w0:
            weight_file.write(" " + str(i))
    return w1,w0

def main():
    args=parse_args()
    w1=None
    w0=None
    if (args.weighted==True):
        if args.w1_file==None:
            w1,w0=get_weights(args.train_path, args.save_weights)
        else:
            w0=[float(i) for i in open(args.w0_file,'r').read().strip().split('\n')]
            w1=[float(i) for i in open(args.w1_file,'r').read().strip().split('\n')]
        print("got weights!")
    try:
        if (args.architecture_from_file!=None):
            architecture_module=imp.load_source('',args.architecture_from_file)
        else:
            architecture_module=importlib.import_module('kerasAC.architectures.'+args.architecture_spec)
    except:
        print("could not import requested architecture, is it installed in kerasAC/kerasAC/architectures? Is the file with the requested architecture specified correctly?")
    model=architecture_module.getModelGivenModelOptionsAndWeightInits(w0,w1,args.init_weights,args.from_checkpoint_weights,args.from_checkpoint_arch,args.num_tasks,args.seed)
    print("compiled the model!")
    train_generator=data_generator(args.train_path,args)
    print("generated training data generator!")
    valid_generator=data_generator(args.valid_path,args)
    print("generated validation data generator!")
    fit_and_evaluate(model,train_generator,
                     valid_generator,args)

if __name__=="__main__":
    main()
