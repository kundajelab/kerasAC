from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from .custom_losses import *
from .metrics import *
from keras.models import load_model
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from abstention.calibration import PlattScaling, IsotonicRegression 
import argparse
def parse_args():
    parser=argparse.ArgumentParser(description="calibration of model preacts/logits")
    parser.add_argument("--preacts",help="preact/logit hdf5 file generated by kerasAC_predict")
    parser.add_argument("--labels",help="hdf5 file generated by kerasAC_predict")
    parser.add_argument("--model",help="hdf5 file generated by kerasAC_train")
    parser.add_argument("--outf",help="name of output hdf5 file")
    parser.add_argument("--calibrate_regression",action="store_true",default=False)
    parser.add_argument("--calibrate_classification",action="store_true",default=False) 
    return parser.parse_args() 

def load_model_wrapper(model_hdf5_fname):
    custom_objects={"recall":recall,
                    "sensitivity":recall,
                    "specificity":specificity,
                    "fpr":fpr,
                    "fnr":fnr,
                    "precision":precision,
                    "f1":f1,
                    "ambig_binary_crossentropy":ambig_binary_crossentropy,
                    "ambig_mean_absolute_error":ambig_mean_absolute_error,
                    "ambig_mean_squared_error":ambig_mean_squared_error,
                    "MultichannelMultinomialNLL":MultichannelMultinomialNLL}
    get_custom_objects().update(custom_objects)
    model=load_model(model_hdf5_fname)
    return model

def get_preacts(model,calibrate_classification=False, calibrate_regression=False):
    if calibrate_classification==True:
        print("getting logits")
        return Model(inputs=model.input,
                     outputs=model.layers[-2].output)
    elif calibrate_regression==True:
        print("getting pre-relu outputs (preacts)")
        return Model(inputs=model.input,
                    outputs=model.layers[-1].output)


def calibrate(preacts,labels,model,outf,calibrate_regression=False,calibrate_classification=False,get_model_preacts=False):
    assert not ((calibrate_classification==False) and (calibrate_regression==False))
    assert not ((calibrate_classification==True) and (calibrate_regression==True))
    if(type(preacts)!=type(pd.DataFrame)):
        #load the preacts
        preacts=pd.read_hdf(preacts)
        print("loaded preacts from hdf5")
    if(type(labels)!=type(pd.DataFrame)):
        labels=pd.read_hdf(labels)
        print("loaded labels from hdf5") 
    #make sure they are in the same order
    labels=labels.loc[preacts.index]
    print("ordered labels")
    if type(model)==str:
        model=load_model_wrapper(model)
        get_model_preact=True 
    if get_model_preact==True:
        model=get_preacts(model,calibrate_classification=calibrate_classification, calibrate_regression=calibrate_regression)
        
        
    #perform calibration for each task!
    calibrated_predictions=None
    for i in range(preacts.shape[1]):
        #don't calibrate on nan inputs
        nonambiguous_indices=np.argwhere(~np.isnan(labels[i]))
        if calibrate_classification==True:
            calibration_func = PlattScaling()(valid_preacts=preacts[i][nonambiguous_indices],
                                                             valid_labels=labels[i][nonambiguous_indices])
        elif calibrate_regression==True:
            calibration_func=IsotonicRegression()(valid_preacts=preacts[i][nonambiguous_indices].squeeze(),
                                                  valid_labels=labels[i][nonambiguous_indices].squeeze())

        calibrated_predictions_task=calibration_func(preacts[i].values)
        if calibrated_predictions is None:
            calibrated_predictions=np.expand_dims(calibrated_predictions_task,axis=1)
        else:
            calibrated_predictions=np.concatenate((calibrated_predictions,np.expand_dims(calibrated_predictions_task,axis=1)),axis=1)
    calibrated_predictions=pd.DataFrame(calibrated_predictions,index=preacts.index)
    calibrated_predictions.to_hdf(outf,key="data",mode='w',append=False,format="table",min_itemsize={'CHR':30})

def main():     
    args=parse_args()
    calibrate(args.preacts,args.labels,args.model,args.outf,calibrate_regression=args.calibrate_regression, calibrate_classification=args.calibrate_classification,get_model_preacts=True)

    
if __name__=="__main__":
    main()