import kerasAC.metrics
import kerasAC.activations 
from kerasAC.accuracy_metrics import *
import argparse
import yaml 
import h5py 
import pickle
import numpy as np 
import keras 
from keras.losses import *
from kerasAC.custom_losses import * 
import random
from .generators import *
from .config import args_object_from_args_dict


def get_predictions_hammock(args,model):
    import pysam
    import pandas as pd
    num_generated=0
    ref=pysam.FastaFile(args.ref_fasta) 
    data=pd.read_csv(args.data_hammock,header=None,sep='\t')
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    #iterate through batches and one-hot-encode on the fly
    num_entries=data.shape[0]
    predictions=None
    all_names=[] 
    while num_generated < num_entries:
        print(str(num_generated))
        start_index=num_generated
        end_index=min([num_entries,start_index+args.batch_size])
        seqs=[]
        names=[]
        for i in range(start_index,end_index):
            cur_row=data.iloc[i]
            chrom=cur_row[0]
            start_val=cur_row[1]
            end_val=cur_row[2]
            peak_metadata=cur_row[3].split(',') 
            peak_name=','.join([peak_metadata[-3],peak_metadata[-2]])
            if args.center_on_summit==True:
                summit_offset=int(peak_metadata[-1].split('[')[1].split(']')[0])
                summit_pos=start_val+summit_offset
                start_val=summit_pos - args.flank
                end_val=summit_pos+args.flank
            if start_val<1:
                start_val=1
                end_val=1+2*args.flank 
            peak_name='\t'.join([str(i) for i in [chrom,start_val,end_val,peak_name]])
            names.append(peak_name)
            try:
                seq=ref.fetch(chrom,start_val,end_val)
            except:
                continue
            seqs.append(seq)
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        if (args.squeeze_input_for_gru==False):
            #expand dimension of 1
            x=np.expand_dims(seqs,1)
        else:
            x=seqs
        try:
            predictions_batch=model.predict(x)
        except:
            print("could not get predictions -- chances are reference assembly is wrong, or bed region lies outside of chrom sizes") 
        #add the batch predictions to the full set of predictions
        if type(predictions)==type(None):
            predictions=predictions_batch
        elif type(predictions)==np.ndarray:
            predictions=np.concatenate((predictions,predictions_batch),axis=0)
            print(predictions.shape)
        elif type(predictions)==type({}):
            for key in predictions_batch:
                predictions[key]=np.concatenate((predictions[key],predictions_batch[key]),axis=0)
        else:
            print("Unsupported data type for predictions: must be np.ndarray, None, or dictionary")
            pdb.set_trace()
        all_names=all_names+names 
        num_generated+=(end_index-start_index)
        
    return [predictions,data,all_names]

def get_predictions(args,model):
    import pysam
    import pandas as pd
    test_generator=DataGenerator(args.data_path,args.ref_fasta,upsample=False,add_revcomp=False,batch_size=1000,chroms_to_use=args.predict_chroms)
    test_predictions=model.predict_generator(test_generator,
                                             max_queue_size=args.max_queue_size,
                                             workers=args.workers,
                                             use_multiprocessing=True,
                                             verbose=1)
    return [predictions,test_generator.data,None]

def get_predictions_variant(args,model):
    import pysam
    ref=pysam.FastaFile(args.ref_fasta)
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
    parser.add_argument('--data_hdf5',help='hdf5 file that stores the data')
    parser.add_argument('--data_path',required=True)
    parser.add_argument('--predict_chroms',default=None) 
    parser.add_argument('--data_hammock',help='input file is in hammock format, with unique id for each peak')
    parser.add_argument('--variant_bed')
    parser.add_argument('--predictions_pickle',help='name of pickle to save predictions',required=True)
    parser.add_argument('--accuracy_metrics_file',help='file name to save accuracy metrics',required=True)
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
    parser.add_argument('--sequential',default=False,help='use this flag if your model is a sequential model',action="store_true")
    parser.add_argument('--ref_fasta',default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
    parser.add_argument('--background_freqs',default=None)
    parser.add_argument('--w1',nargs="*",type=float)
    parser.add_argument('--w0',nargs="*",type=float)
    parser.add_argument('--flank',default=500,type=int)
    parser.add_argument('--mask',default=10,type=int)
    parser.add_argument('--squeeze_input_for_gru',action='store_true')
    parser.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    return parser.parse_args()

def get_model(args):
    custom_objects={"positive_accuracy":kerasAC.metrics.positive_accuracy,
                    "negative_accuracy":kerasAC.metrics.negative_accuracy,
                    "precision":kerasAC.metrics.precision,
                    "recall":kerasAC.metrics.recall,
                    "softMaxAxis1":kerasAC.activations.softMaxAxis1}
    if args.w0!=None:
        w0=args.w0
        w1=args.w1
        loss_function=get_weighted_binary_crossentropy(w0,w1)
        custom_objects["weighted_binary_crossentropy"]=loss_function
    try:
        if args.yaml!=None:
            from keras.models import model_from_yaml
            #load the model architecture from yaml
            yaml_string=open(args.yaml,'r').read()
            model=model_from_yaml(yaml_string,custom_objects=custom_objects) 
            #load the model weights
            model.load_weights(args.weights)
        elif args.json!=None:
            from keras.models import model_from_json
            #load the model architecture from json
            json_string=open(args.json,'r').read()
            model=model_from_json(json_string,custom_objects=custom_objects)
            model.load_weights(args.weights)
        elif args.model_hdf5!=None: 
            #load from the hdf5
            from keras.models import load_model
            model=load_model(args.model_hdf5,custom_objects=custom_objects)
        print("got model architecture")
        print("loaded model weights")        
        
    except: 
        print("Failed to load model. HINT: if you're using weighted binary cross entropy loss, chances are you forgot to provide the --w0 or --w1 flags")
    return model

def get_predictions(args,model):
    if args.variant_bed!=None:
        predictions=get_predictions_variant(args,model)
    elif args.data_hammock!=None:
        predictions=get_predictions_hammock(args,model) 
    else:
        predictions=get_predictions(args,model) 
    print('got model predictions')
    return predictions

def predict(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    
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


def main():
    args=parse_args()
    predict(args) 

    

if __name__=="__main__":
    main()
    
