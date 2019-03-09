from kerasAC.activations import softMaxAxis1
from kerasAC.generators import *
from kerasAC.config import args_object_from_args_dict
from kerasAC.performance_metrics import *
from kerasAC.custom_losses import *
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1
import argparse
import yaml 
import h5py 
import pickle
import numpy as np 
import keras 
from keras.losses import *
from kerasAC.custom_losses import *
from abstention.calibration import PlattScaling, IsotonicRegression 
import random
from scipy.special import logit,expit
import pdb 

def get_weights(args):
    w1=None
    w0=None
    if args.w1_w0_file!=None:
        w1_w0=np.loadtxt(args.w1_w0_file)
        w1=w1_w0[:,0]
        w0=w1_w0[:,1]
    if args.w1!=None:
        w1=args.w1
    if args.w0!=None:
        w0=args.w0 
    return w1,w0

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

def get_predictions_basic(args,model):
    import pysam
    import pandas as pd
    test_generator=DataGenerator(args.data_path,args.ref_fasta,upsample=False,add_revcomp=False,batch_size=1000,chroms_to_use=args.predict_chroms)
    predictions=model.predict_generator(test_generator,
                                        max_queue_size=args.max_queue_size,
                                        workers=args.threads,
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
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--max_queue_size",type=int,default=100)
    parser.add_argument('--model_hdf5',help='hdf5 file that stores the model')
    parser.add_argument('--weights',help='weights file for the model')
    parser.add_argument('--yaml',help='yaml file for the model')
    parser.add_argument('--json',help='json file for the model')
    parser.add_argument('--data_path',required=True)
    parser.add_argument('--predict_chroms',nargs="*",default=None) 
    parser.add_argument('--data_hammock',help='input file is in hammock format, with unique id for each peak')
    parser.add_argument('--variant_bed')
    parser.add_argument('--predictions_pickle',help='name of pickle to save predictions',default=None)
    parser.add_argument('--performance_metrics_classification_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--performance_metrics_regression_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
    parser.add_argument('--functional',default=False,help='use this flag if your model is a functional model',action="store_true")
    parser.add_argument('--ref_fasta',default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
    parser.add_argument('--background_freqs',default=None)
    parser.add_argument('--w1',nargs="*",type=float)
    parser.add_argument('--w0',nargs="*",type=float)
    parser.add_argument("--w1_w0_file",default=None)
    parser.add_argument('--flank',default=500,type=int)
    parser.add_argument('--mask',default=10,type=int)
    parser.add_argument('--squeeze_input_for_gru',action='store_true')
    parser.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    parser.add_argument("--calibrate_classification",action="store_true",default=False)
    parser.add_argument("--calibrate_regression",action="store_true",default=False) 
    return parser.parse_args()

def get_model(args):

    from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1
    
    custom_objects={"recall":recall,
                    "sensitivity":recall,
                    "specificity":specificity,
                    "fpr":fpr,
                    "fnr":fnr,
                    "precision":precision,
                    "f1":f1,
                    "ambig_binary_crossentropy":get_ambig_binary_crossentropy(),
                    "ambig_mean_squared_error":get_ambig_mean_squared_error()}
    w1,w0=get_weights(args)
    if type(w1) in [np.ndarray, list]: 
        loss_function=get_weighted_binary_crossentropy(w0,w1)
        custom_objects["weighted_binary_crossentropy"]=loss_function
#    try:
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
        
#    except: 
#        print("Failed to load model. HINT: if you're using weighted binary cross entropy loss, chances are you forgot to provide the --w0 or --w1 flags")
    return model

def get_predictions(args,model):
    if args.variant_bed!=None:
        predictions=get_predictions_variant(args,model)
    elif args.data_hammock!=None:
        predictions=get_predictions_hammock(args,model) 
    else:
        predictions=get_predictions_basic(args,model) 
    print('got model predictions')
    return predictions

def calibrate(predictions,args):
    if args.calibrate_classification == True:
        #calibrate classification predictions
        #avoid -inf values and inf values:
        pseudocount=1e-5
        logit_input=predictions[0]
        logit_input[logit_input<=0]+=pseudocount
        logit_input[logit_input>=1]-=pseudocount 
        logits=logit(logit_input)
        labels=predictions[1].values
        #perform calibration for each task!
        calibrated_predictions=None
        for i in range(logits.shape[1]):
            #don't calibrate on nan inputs
            nonambiguous_indices=np.argwhere(~np.isnan(labels[:,i]))
            classification_calibration_func = PlattScaling()(
                valid_preacts=logits[nonambiguous_indices,i],
                valid_labels=labels[nonambiguous_indices,i])
            calibrated_predictions_task=classification_calibration_func(logits[:,i])
            if calibrated_predictions is None:
                calibrated_predictions=np.expand_dims(calibrated_predictions_task,axis=1)
            else:
                calibrated_predictions=np.concatenate((calibrated_predictions,np.expand_dims(calibrated_predictions_task,axis=1)),axis=1)
        predictions.append(logits)
        predictions.append(calibrated_predictions)        
        print("predictions calibrated with Platt scaling")

    elif args.calibrate_regression==True:
        regression_calibration_func=IsotonicRegression()(
            valid_preacts=predictions[0],
            valid_labels=predictions[1].values)
        calibrated_predictions=regression_calibration_func(predictions[0])
        predictions.append(None)
        predictions.append(calibrated_predictions)
        print("predictions calibrated with Isotonic Regression")
    return predictions

def predict(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    
    #get the predictions
    if args.predictions_pickle_to_load!=None:
        #load the pickled predictions
        with open(args.predictions_pickle_to_load,'rb') as handle:
            predictions=pickle.load(handle)
        print("loaded predictions from pickle") 
    else:
        #get the model
        model=get_model(args)
        predictions=get_predictions(args,model)
        assert not ((args.calibrate_classification==True) and (args.calibrate_regression==True))
        if args.predictions_pickle!=None:
            #pickle the predictions in case an error occurs downstream
            #this will allow for easy recovery of model predictions without having to regenerate them
            with open(args.predictions_pickle,'wb') as handle:
                pickle.dump(predictions,handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("pickled the model predictions to file:"+str(args.predictions_pickle))
        predictions=calibrate(predictions,args)
    
    if ((args.performance_metrics_classification_file!=None) or (args.performance_metrics_regression_file!=None)):
        labels=predictions[1].values 
        tasks=predictions[1].columns 
        #if calibration has been used, we want accuracy metrics on the calibrated predictions (last entry in predictions list) 
        if ((args.calibrate_classification==True) or (args.calibrate_regression==True)): 
            predictions=predictions[-1]
        else:
            predictions=predictions[0]
            
        if args.performance_metrics_classification_file!=None:
            print("calculating classification performance metrics...")
            performance_metrics_classification=get_performance_metrics_classification(predictions,labels)
            print("writing classification performance metrics to file...") 
            write_performance_metrics(args.performance_metrics_classification_file,performance_metrics_classification,tasks) 
        elif args.performance_metrics_regression_file!=None:
            print("calculating regression performance metrics...") 
            performance_metrics_regression=get_performance_metrics_regression(predictions,labels)
            print("writing regression performance metrics to file...") 
            write_performance_metrics(args.performance_metrics_regression_file,performance_metrics_regression,tasks)
            
#write performance metrics to output file: 
def write_performance_metrics(output_file,metrics_dict,tasks):
    metrics_df=pd.DataFrame(metrics_dict,index=tasks).transpose()
    metrics_df.to_csv(output_file,sep='\t',header=True,index=True)
    
def main():
    args=parse_args()
    predict(args) 

    

if __name__=="__main__":
    main()
    
