import argparse
import yaml 
import h5py 
import keras
from accuracy_metrics import * 
import pickle
import numpy as np 
import pdb
import keras 
from keras.losses import *;
import random

def get_predictions_hdf5(args,model):
    data=h5py.File(args.data,'r')    
    if args.sequential==True: 
        inputs=data['X']['default_input_mode_name']
        outputs=data['Y']['default_output_mode_name']
        output_task_names=None
        individual_task_output_shape=outputs.shape
        predictions=get_predictions_hdf5_sequenctial(data,args.batch_size,individual_task_output_shape,output_task_names,model)[0]
    else:
        inputs=data['X']
        outputs=data['Y'] 
        output_task_names=outputs.keys()
        individual_task_output_shape=outputs.values()[0].shape
        predictions= get_preditions_hdf5_functional(data,args.batch_size,individual_task_output_shape,output_task_names,model)[0]
    return [predictions,outputs]

def get_predictions_hdf5_functional(hdf5_source,batch_size,individual_task_output_shape,output_task_names,model):
    if output_task_names==None:
        return get_predictions_sequential(hdf5_source,batch_size,individual_task_output_shape,model)
    num_generated=0
    total_entries=hdf5_source.values()[0].shape[0]

    input_modes=hdf5_source.keys() 
    print("total entries:"+str(total_entries))
    predictions={}
    for task in output_task_names:
        predictions[task]=np.zeros(individual_task_output_shape)
    print("initialized output dictionary for predictions")    
    while num_generated < total_entries:
        print(str(num_generated))
        start_index=num_generated
        end_index=min([total_entries,start_index+batch_size])
        x_batch={}  
        for input_mode in input_modes: 
            x_batch[input_mode] = hdf5_source[input_mode][start_index:end_index,:,:,500:1500]
            x_batch=np.transpose(x_batch,axes=(0,1,3,2))
        predictions_batch=model.predict(x_batch)
        #add the predictions to the dictionary
        for task in output_task_names:
            predictions[task][start_index:end_index]=predictions_batch[task]
        num_generated+=(end_index-start_index)
    return [predictions,None]

def get_predictions_sequential_hdf5(hdf5_source,batch_size,individual_task_output_shape,model):
    num_generated=0
    total_entries=hdf5_source.shape[0]
    print("total entries:"+str(total_entries))
    predictions=np.zeros(individual_task_output_shape)
    print("initialized output dictionary for predictions")    
    while num_generated < total_entries:
        print(str(num_generated)) 
        start_index=num_generated
        end_index=min([total_entries,start_index+batch_size])
        x_batch=hdf5_source[start_index:end_index,:,:,500:1500]
        x_batch=np.transpose(x_batch,axes=(0,1,3,2))
        predictions_batch=model.predict(x_batch)
        #add the predictions to the dictionary
        predictions[start_index:end_index]=predictions_batch
        num_generated+=(end_index-start_index)
    return [predictions,None]

def get_predictions_bed(args,model):
    import pysam
    import pandas as pd
    num_generated=0
    ref=pysam.FastaFile(args.ref)
    data=pd.read_csv(args.data_bed,header=0,sep='\t',index_col=[0,1,2])
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    #iterate through batches and one-hot-encode on the fly
    num_entries=data.shape[0]
    bed_entries=[(data.index[i]) for i in range(num_entries)]
    seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
    seqs=np.array([[ltrdict[x] for x in seq] for seq in seqs])
    #expand dimension of 1
    x=np.expand_dims(seqs,1)
    print(x.shape) 
    #y=np.asarray(data)
    #print(y.shape)
    try:
        predictions=model.predict(x)
    except:
        pdb.set_trace()
    return [predictions,data]

def get_predictions_variant(args,model):
    import pysam
    ref=pysam.FastaFile(args.ref)
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    #assumes the bed file has `chrom start ref alt` entries
    data=[i.split('\t') for i in open(args.variant_bed,'r').read().strip().split('\n')]
    #original
    seqs=[]
    #snp
    seqs_snp=[]
    #n bases around snp (i.e. knock out an enhancer)
    seqs_enhancer=[] 
    for entry in data[1::]:
        #introduce the specified variant 
        start_pos=int(entry[1])-(args.flank)
        end_pos=int(entry[1])+args.flank
        seq=ref.fetch(entry[0],start_pos,end_pos)
        seqs.append(seq)
    
        
        alt_allele=entry[3]
        if alt_allele=="NA":
            #randomly insert another base
            ref_allele=seq[args.flank-1].upper()
            options=['A','C','G','T']
            options.remove(ref_allele)
            alt_allele=options[random.randint(0,2)]
        seq=seq[0:args.flank-1]+alt_allele+seq[args.flank:len(seq)]
        seqs_snp.append(seq)

        seq=np.array([ltrdict[x] for x in seq])
        start_mask=args.flank-1-args.mask
        end_mask=args.flank-1+args.mask
        seq=seq*1.0
        seq[start_mask:end_mask]=args.background_freqs
        seqs_enhancer.append(seq)
        
    seqs=np.array([[ltrdict[x] for x in seq] for seq in seqs])    
    seqs_snp=np.array([[ltrdict[x] for x in seq] for seq in seqs_snp])
    seqs_enhancer=np.asarray(seqs_enhancer)

    seqs=np.expand_dims(seqs,1)
    seqs_snp=np.expand_dims(seqs_snp,1)
    seqs_enhancer=np.expand_dims(seqs_enhancer,1)
    
    predictions_seqs=model.predict(seqs)
    predictions_seqs_snp=model.predict(seqs_snp)
    predictions_seqs_enhancer=model.predict(seqs_enhancer)
    return [predictions_seqs,predictions_seqs_snp,predictions_seqs_enhancer]

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model yaml & weights files & a dataset, get model predictions and accuracy metrics')
    parser.add_argument('--model_hdf5',help='hdf5 file that stores the model')
    parser.add_argument('--weights',help='weights file for the model')
    parser.add_argument('--yaml',help='yaml file for the model')
    parser.add_argument('--json',help='json file for the model')
    parser.add_argument('--data_hdf5',help='hdf5 file that stores the data -- validation hdf5 file')
    parser.add_argument('--data_bed')
    parser.add_argument('--variant_bed')
    parser.add_argument('--predictions_pickle',help='name of pickle to save predictions',default=None)
    parser.add_argument('--accuracy_metrics_file',help='file name to save accuracy metrics',default=None)
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
    parser.add_argument('--sequential',default=False,help='use this flag if your model is a sequential model',action="store_true")
    parser.add_argument('--ref',default=None)
    parser.add_argument('--background_freqs',default=None)
    parser.add_argument('--w1',nargs="*",type=float)
    parser.add_argument('--w0',nargs="*",type=float)
    parser.add_argument('--flank',default=500,type=int)
    parser.add_argument('--mask',default=10,type=int) 
    return parser.parse_args()

def get_model(args):
    if args.model_hdf5==None:
        yaml_string=open(args.yaml,'r').read()
        model_config=yaml.load(yaml_string)
        model=Graph.from_config(model_config)
        print("got model architecture")
        #load the model weights
        model.load_weights(args.weights)
        print("loaded model weights")
    else:
        #load from the hdf5
        from keras.models import load_model
        if args.w0!=None:
            w0=args.w0
            w1=args.w1
            loss_function=get_weighted_binary_crossentropy(w0,w1)                
            model=load_model(args.model_hdf5,custom_objects={"weighted_binary_crossentropy":loss_function})
        else:
            model=load_model(args.model_hdf5)
    return model

def get_predictions(args,model):
    if args.data_hdf5!=None:
        predictions=get_predictions_hdf5(args,model)
    elif args.data_bed!=None:
        predictions=get_predictions_bed(args,model)
    elif args.variant_bed!=None:
        predictions=get_predictions_variant(args,model)
    else:
        raise Exception("input data must be specified by data_hdf5, data_bed, or data_variant")
    print('got model predictions')
    return predictions

def main():
    args=parse_args()
    
    #get the predictions
    if args.predictions_pickle_to_load!=None:
        #load the pickled predictions
        with open(args.predictions_pickle_to_load,'rb') as handle:
            predictions=pickle.load(handle)
    else:
        #get the model
        model=get_model(args)
        predictions=get_predictions(args,model)
        
        
    if args.predictions_pickle!=None:
        #pickle the predictions in case an error occurs downstream
        #this will allow for easy recovery of model predictions without having to regenerate them
        with open(args.predictions_pickle,'wb') as handle:
            pickle.dump(predictions,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("pickled the model predictions to file:"+str(args.predictions_pickle))

    if args.accuracy_metrics_file!=None:
        print('computing accuracy metrics...')
        recallAtFDR50=dict()
        recallAtFDR20=dict()
        auroc_vals=dict()
        auprc_vals=dict()
        unbalanced_accuracy_vals=dict()
        balanced_accuracy_vals=dict()
        positives_accuracy_vals=dict()
        negatives_accuracy_vals=dict()
        num_positive_vals=dict()
        num_negative_vals=dict()
        
        outputs=np.asarray(predictions[1])
        predictions=predictions[0]
        if not(isinstance(predictions,dict)):
            predictions={'output':predictions}
        if not(isinstance(outputs,dict)):
            outputs={'output':outputs} 
        for output_mode in predictions: 
            #compute the accuracy metrics
            recallAtFDR50[output_mode]=recall_at_fdr_function(predictions[output_mode],outputs[output_mode],50)
            print('got recall at FDR50!') 
            recallAtFDR20[output_mode]=recall_at_fdr_function(predictions[output_mode],outputs[output_mode],20)
            print('got recall at FDR20!')
            auroc_vals[output_mode]=auroc_func(predictions[output_mode],outputs[output_mode])
            print('got auROC vals!')
            auprc_vals[output_mode]=auprc_func(predictions[output_mode],outputs[output_mode])
            print('got auPRC vals!')
            unbalanced_accuracy_vals[output_mode]=unbalanced_accuracy(predictions[output_mode],outputs[output_mode])
            print('got unbalanced accuracy')
            balanced_accuracy_vals[output_mode]=balanced_accuracy(predictions[output_mode],outputs[output_mode])
            print('got balanced accuracy')
            positives_accuracy_vals[output_mode]=positives_accuracy(predictions[output_mode],outputs[output_mode])
            print('got positives accuracy')
            negatives_accuracy_vals[output_mode]=negatives_accuracy(predictions[output_mode],outputs[output_mode])
            print('got negative accuracy vals')
            num_positive_vals[output_mode]=num_positives(predictions[output_mode],outputs[output_mode])
            print('got number of positive values')
            num_negative_vals[output_mode]=num_negatives(predictions[output_mode],outputs[output_mode])
            print('got number of negative values')

        #write accuracy metrics to output file: 
        print('writing accuracy metrics to file...')
        outf=open(args.accuracy_metrics_file,'w')
        for key in recallAtFDR50.keys():
            outf.write('recallAtFDR50\t'+str(key)+'\t'+'\t'.join([str(i) for i in recallAtFDR50[key]])+'\n')
        for key in recallAtFDR20.keys():
            outf.write('recallAtFDR20\t'+str(key)+'\t'+'\t'.join([str(i) for i in recallAtFDR20[key]])+'\n')
        for key in auroc_vals.keys():
            outf.write('auroc_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in auroc_vals[key]])+'\n')
        for key in auprc_vals.keys():
            outf.write('auprc_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in auprc_vals[key]])+'\n')
        for key in unbalanced_accuracy_vals.keys():
            outf.write('unbalanced_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in unbalanced_accuracy_vals[key]])+'\n')
        for key in balanced_accuracy_vals.keys():
            outf.write('balanced_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in balanced_accuracy_vals[key]])+'\n')
        for key in positives_accuracy_vals.keys():
            outf.write('positives_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in positives_accuracy_vals[key]])+'\n')
        for key in negatives_accuracy_vals.keys():
            outf.write('negatives_accuracy_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in negatives_accuracy_vals[key]])+'\n')    
        for key in num_positive_vals.keys():
            outf.write('num_positive_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in num_positive_vals[key]])+'\n')
        for key in num_negative_vals.keys():
            outf.write('num_negative_vals\t'+str(key)+'\t'+'\t'.join([str(i) for i in num_negative_vals[key]])+'\n')
    

if __name__=="__main__":
    main()
    
