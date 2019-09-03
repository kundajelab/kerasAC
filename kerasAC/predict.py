from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from keras import callbacks as cbks
from kerasAC.activations import softMaxAxis1
from kerasAC.generators import *
from kerasAC.tiledb_predict import * 
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
from keras.models import Model 
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

def get_predictions_tiledb(args,model):
    import pysam
    import pandas as pd
    test_generator=TiledbPredictGenerator(batch_size=args.batch_size,
                                          task_file=args.tiledb_tasks_file,
                                          ref_fasta=args.ref_fasta,
                                          label_source=args.label_source_attribute,
                                          label_flank=args.label_flank,
                                          label_aggregation=args.label_aggregation,
                                          sequence_flank=args.sequence_flank,
                                          tiledb_stride=args.tiledb_stride,
                                          chrom_sizes_file=args.chrom_sizes,
                                          chroms=args.predict_chroms)
    
    predictions=model.predict_generator(test_generator,
                                        max_queue_size=args.max_queue_size,
                                        workers=args.threads,
                                        use_multiprocessing=True,
                                        verbose=1)
    print("got predictions")
    #iterate through to generator to get coords and labels
    
def get_predictions_bed(args,model):
    import pysam
    import pandas as pd
    test_generator=DataGenerator(data_path=args.data_path,
                                 nonzero_bin_path=args.nonzero_bin_path,
                                 universal_negative_path=args.universal_negative_path,
                                 ref_fasta=args.ref_fasta,
                                 batch_size=args.batch_size,
                                 add_revcomp=False,
                                 chroms_to_use=args.predict_chroms,
                                 expand_dims=args.expand_dims,
                                 tasks=args.tasks,
                                 shuffle=False)
    predictions=model.predict_generator(test_generator,
                                  max_queue_size=args.max_queue_size,
                                  workers=args.threads,
                                  use_multiprocessing=True,
                                  verbose=1)
    print("got predictions")
    perform_calibration=args.calibrate_classification or args.calibrate_regression
    preacts=None
    if perform_calibration==True:
        if args.calibrate_classification==True:
            print("getting logits")
            preact_model=Model(inputs=model.input,
                               outputs=model.layers[-2].output)
        elif args.calibrate_regression==True:
            print("getting pre-relu outputs (preacts)")
            preact_model=Model(inputs=model.input,
                              outputs=model.layers[-1].output) 
        test_generator=DataGenerator(args.data_path,
                                     nonzero_bin_path=args.nonzero_bin_path,
                                     universal_negative_path=args.universal_negative_path,
                                     ref_fasta=args.ref_fasta,
                                     add_revcomp=False,
                                     batch_size=args.batch_size,
                                     chroms_to_use=args.predict_chroms,
                                     expand_dims=args.expand_dims,
                                     tasks=args.tasks,
                                     shuffle=False)
        
        preacts=preact_model.predict_generator(test_generator,
                                  max_queue_size=args.max_queue_size,
                                  workers=args.threads,
                                  use_multiprocessing=True,
                                  verbose=1)
        print("preactivation shape:"+str(preacts.shape))
    return [predictions,test_generator.get_labels(),preacts]

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
    
    input_data_path=parser.add_argument_group("input_data_path")
    input_data_path.add_argument('--data_path',default=None,help="bed regions, either as tsv or pandas")
    input_data_path.add_argument("--nonzero_bin_path",default=None)
    input_data_path.add_argument("--universal_negative_path",default=None)
    input_data_path.add_argument('--variant_bed',default=None)
    input_data_path.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    input_data_path.add_argument('--tiledb_tasks_file',default=None,help="path to tiledb database")
    input_data_path.add_argument('--ref_fasta')

    tiledb_group=parser.add_argument_group("tiledb")
    tiledbgroup.add_argument("--chrom_sizes",default=None,help="chromsizes file for use with tiledb generator")
    tiledbgroup.add_argument("--label_source_attribute",default="fc_bigwig",help="tiledb attribute for use in label generation i.e. fc_bigwig")
    tiledbgroup.add_argument("--label_flank",type=int,help="flank around bin center to use in generating labels")
    tiledbgroup.add_argument("--label_aggregation",default=None,help="one of None, 'avg','max'")
    tiledbgroup.add_argument("--sequence_flank",type=int,help="length of sequence around bin center to use in one-hot-encoding")
    tiledbgroup.add_argument("--tiledb_stride",type=int,default=1)

    input_filtering_params=parser.add_argument_group("input_filtering_params")    
    input_filtering_params.add_argument('--predict_chroms',nargs="*",default=None)
    input_filtering_params.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    input_filtering_params.add_argument("--tasks",nargs="*",default=None)
    
    output_params=parser.add_argument_group("output_params")
    output_params.add_argument('--predictions_pickle',help='name of pickle to save predictions',default=None)
    parser.add_argument('--performance_metrics_classification_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--performance_metrics_regression_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)

    calibration_params=parser.add_argument_group("calibration_params")
    calibration_params.add_argument("--calibrate_classification",action="store_true",default=False)
    calibration_params.add_argument("--calibrate_regression",action="store_true",default=False)        
    
    weight_params=parser.add_argument_group("weight_params")
    weight_params.add_argument('--w1',nargs="*",type=float)
    weight_params.add_argument('--w0',nargs="*",type=float)
    weight_params.add_argument("--w1_w0_file",default=None)


    model_params=parser.add_argument_group("model_params")
    model_params.add_argument('--model_hdf5',help='hdf5 file that stores the model')
    model_params.add_argument('--weights',help='weights file for the model')
    model_params.add_argument('--yaml',help='yaml file for the model')
    model_params.add_argument('--json',help='json file for the model')
    model_params.add_argument('--functional',default=False,help='use this flag if your model is a functional model',action="store_true")
    model_params.add_argument('--squeeze_input_for_gru',action='store_true')
    model_params.add_argument("--expand_dims",default=True)
    
    parallelization_params=parser.add_argument_group("parallelization")
    parallelization_params.add_argument("--threads",type=int,default=1)
    parallelization_params.add_argument("--max_queue_size",type=int,default=100)

    snp_params=parser.add_argument_group("snp_params")
    snp_params.add_argument('--background_freqs',default=None)
    snp_params.add_argument('--flank',default=500,type=int)
    snp_params.add_argument('--mask',default=10,type=int)

    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
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
                    "ambig_binary_crossentropy":ambig_binary_crossentropy,
                    "ambig_mean_absolute_error":ambig_mean_absolute_error,
                    "ambig_mean_squared_error":ambig_mean_squared_error}
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
    if args.variant_bed is not None:
        predictions=get_predictions_variant(args,model)
    elif args.tiledb_tasks_file is not None:
        predictions=get_predictions_tiledb(args,model)
    else:
        predictions=get_predictions_bed(args,model) 
    print('got model predictions')
    return predictions


def get_model_layer_functor(model,target_layer_idx):
    from keras import backend as K
    inp=model.input
    outputs=model.layers[target_layer_idx].output
    functor=K.function([inp], [outputs])
    return functor 

def get_layer_outputs(functor,X):
    return functor([X])

def calibrate(predictions,args,model):
    preacts=predictions[2]
    labels=predictions[1].values
    #perform calibration for each task!
    calibrated_predictions=None
    for i in range(preacts.shape[1]):
        #don't calibrate on nan inputs
        nonambiguous_indices=np.argwhere(~np.isnan(labels[:,i]))
        if args.calibrate_classification==True:
            calibration_func = PlattScaling()(valid_preacts=preacts[nonambiguous_indices,i],
                                                             valid_labels=labels[nonambiguous_indices,i])
        elif args.calibrate_regression==True:
            calibration_func=IsotonicRegression()(valid_preacts=preacts[nonambiguous_indices,i].squeeze(),
                                                  valid_labels=labels[nonambiguous_indices,i].squeeze())
        calibrated_predictions_task=calibration_func(preacts[:,i])
        if calibrated_predictions is None:
            calibrated_predictions=np.expand_dims(calibrated_predictions_task,axis=1)
        else:
            calibrated_predictions=np.concatenate((calibrated_predictions,np.expand_dims(calibrated_predictions_task,axis=1)),axis=1)
    predictions.append(calibrated_predictions)        
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
    if ((args.calibrate_classification==True) or (args.calibrate_regression==True)):
        predictions=calibrate(predictions,args,model)
    if args.predictions_pickle!=None:
        #pickle the predictions in case an error occurs downstream
        #this will allow for easy recovery of model predictions without having to regenerate them
        with open(args.predictions_pickle,'wb') as handle:
            pickle.dump(predictions,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("pickled the model predictions to file:"+str(args.predictions_pickle))    
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
    print(metrics_df)
    
def main():
    args=parse_args()
    predict(args)

    

if __name__=="__main__":
    main()
    
