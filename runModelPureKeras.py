#Raw keras model 
import imp
import argparse
import numpy as np 
import h5py 
from create_generators import * 

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
    parser.add_argument("--from_checkpoint_weights",default=None)
    parser.add_argument("--from_checkpoint_arch",default=None)
    parser.add_argument("--num_tasks",type=int)
    #add functionality to train on individuals' allele frequencies
    parser.add_argument("--vcf_file",default=None)
    parser.add_argument("--global_vcf",action="store_true")
    parser.add_argument("--revcomp",action="store_true")
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--patience",type=int,default=5)
    parser.add_argument("--architecture_spec",type=str,default="basset_architecture_single_task.py")
    parser.add_argument("--tensorboard",action="store_true")
    parser.add_argument("--tensorboard_logdir",default="logs")
    
    return parser.parse_args() 
        
def fit_and_evaluate(model,train_gen,valid_gen,args):
    model_output_path = args.model_output_file
    from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1)
    csvlogger = CSVLogger(args.model_output_file+".log", append = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=5, min_lr=0.00000001)
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
    
def get_weights(bed_path):
    import pandas as pd
    data=pd.read_csv(bed_path,header=0,sep='\t',index_col=[0,1,2])
    w1=[float(data.shape[0])/sum(data.iloc[:,i]==1) for i in range(data.shape[1])]
    w0=[float(data.shape[0])/sum(data.iloc[:,i]==0) for i in range(data.shape[1])]
    print(str(w1))
    print(str(w0))
    return w1,w0

def main():
    args=parse_args()
    if args.w1_file==None: 
        w1,w0=get_weights(args.train_path)
    else:
        w0=[float(i) for i in open(args.w0_file,'r').read().strip().split('\n')]
        w1=[float(i) for i in open(args.w1_file,'r').read().strip().split('\n')]
    print("got weights!")
    architecture_module=imp.load_source('',args.architecture_spec)
    model=architecture_module.getModelGivenModelOptionsAndWeightInits(w0,w1,args.init_weights,args.from_checkpoint_weights,args.from_checkpoint_arch)
    print("compiled the model!")
    train_generator=data_generator(args.train_path,args)
    print("generated training data generator!") 
    valid_generator=data_generator(args.valid_path,args)
    print("generated validation data generator!") 
    fit_and_evaluate(model,train_generator,
                     valid_generator,args) 

if __name__=="__main__":
    main() 
