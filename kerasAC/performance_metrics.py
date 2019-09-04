from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import warnings
import numpy as np
import pysam
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve 
from scipy.stats import spearmanr, pearsonr 
from .util import enum 
from collections import OrderedDict, defaultdict
import sys 
import pdb

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model prediction pickle to compute performance metrics.')    
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the accuracy metrics",default=None)
    parser.add_argument('--performance_metrics_classification_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--performance_metrics_regression_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--performance_metrics_profile_file',help='file name to save accuracy metrics; accuracy metrics not computed if file not provided',default=None)
    parser.add_argument('--profile_model_tasks',nargs="+")
    parser.add_argument('--profile_model_task_indices',nargs="+")
    return parser.parse_args()


def remove_ambiguous_peaks(predictions, true_y,ambig_val=np.nan): 
    indices_to_remove = np.nonzero(np.isnan(true_y))
    true_y_filtered = np.delete(true_y, indices_to_remove)
    predictions_filtered = np.delete(predictions, indices_to_remove)
    return predictions_filtered, true_y_filtered

def auroc_func(predictions_for_task_filtered, true_y_for_task_filtered):
    try:
        task_auroc = roc_auc_score(y_true=true_y_for_task_filtered,
                                   y_score=predictions_for_task_filtered)
    except Exception as e:
        #if there is only one class in the batch of true_y, then auROC cannot be calculated
        print("Could not calculate auROC:")
        print(str(e))
        task_auroc=None 
    return task_auroc

def auprc_func(predictions_for_task_filtered, true_y_for_task_filtered):
    # sklearn only supports 2 classes (0,1) for the auPRC calculation
    try:
        task_auprc=average_precision_score(true_y_for_task_filtered, predictions_for_task_filtered)
    except:
        print("Could not calculate auPRC:")
        print(sys.exc_info()[0])
        task_auprc=None 
    return task_auprc

    

def get_accuracy_stats_for_task(predictions_for_task_filtered, true_y_for_task_filtered, c):
    predictions_for_task_filtered_round = np.array([round(el) for el in predictions_for_task_filtered])
    accuratePredictions = predictions_for_task_filtered_round==true_y_for_task_filtered;

    numPositives_forTask=np.sum(true_y_for_task_filtered==1,axis=0,dtype="float");
    numNegatives_forTask=np.sum(true_y_for_task_filtered==0,axis=0,dtype="float"); 

    accuratePredictions_positives = np.sum(accuratePredictions*(true_y_for_task_filtered==1),axis=0);
    accuratePredictions_negatives = np.sum(accuratePredictions*(true_y_for_task_filtered==0),axis=0);
    unbalancedAccuracy_forTask = (accuratePredictions_positives + accuratePredictions_negatives)/(numPositives_forTask + numNegatives_forTask)

    positiveAccuracy_forTask = accuratePredictions_positives/numPositives_forTask
    negativeAccuracy_forTask = accuratePredictions_negatives/numNegatives_forTask
    balancedAccuracy_forTask= (positiveAccuracy_forTask+negativeAccuracy_forTask)/2;
    returnDict={'unbalanced_accuracy':unbalancedAccuracy_forTask,
                'positive_accuracy':positiveAccuracy_forTask,
                'negative_accuracy':negativeAccuracy_forTask,
                'balanced_accuracy':balancedAccuracy_forTask,
                'num_positives':numPositives_forTask,
                'num_negatives':numNegatives_forTask}
    return returnDict


def recall_at_fdr_function(predictions_for_task_filtered,true_y_for_task_filtered,fdr_thresh_list):
    for fdr_thresh_index in range(len(fdr_thresh_list)):
        if float(fdr_thresh_list[fdr_thresh_index])>1:
            fdr_thresh_list[fdr_thresh_index]=fdr_thresh_list[fdr_thresh_index]/100
            
    precision,recall,class_thresholds=precision_recall_curve(true_y_for_task_filtered,predictions_for_task_filtered)
    fdr=1-precision

    #remove the last values in recall and fdr, as the scipy precision_recall_curve function sets them to 0 automatically
    recall=np.delete(recall,-1)
    fdr=np.delete(fdr,-1)

    #concatenate recall,fdr, thresholds along axis=1
    recall=np.expand_dims(recall,axis=1)
    fdr=np.expand_dims(fdr,axis=1)
    class_thresholds=np.expand_dims(class_thresholds,axis=1)
    data=np.concatenate((recall,fdr,class_thresholds),axis=1)

    #sort by threshold
    data=data[data[:,2].argsort()]

    #get the recall, fdr at each thresh
    recall_thresholds=[]
    class_thresholds=[]
    for fdr_thresh in fdr_thresh_list:
        try:
            data_index=np.max(np.nonzero(data[:,1]<=fdr_thresh))
            recall_thresholds.append(data[data_index,0])
            class_thresholds.append(data[data_index,2])
        except:
            print("No class threshold can give requested fdr <=:"+str(fdr_thresh))
            recall_thresholds.append(np.nan)
            class_thresholds.append(np.nan)
            
    return recall_thresholds, class_thresholds


def get_performance_metrics_classification(predictions,true_y):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    performance_stats=None
    for c in range(num_cols):
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task,true_y_for_task)

        accuracy_stats_task = get_accuracy_stats_for_task(predictions_for_task_filtered, true_y_for_task_filtered, c)
        auprc_task=auprc_func(predictions_for_task_filtered,true_y_for_task_filtered)
        auroc_task=auroc_func(predictions_for_task_filtered,true_y_for_task_filtered)
        recall,class_thresh=recall_at_fdr_function(predictions_for_task_filtered,true_y_for_task_filtered,[50,20,10])
                
        if performance_stats==None:
            performance_stats=dict()
            for key in accuracy_stats_task:
                performance_stats[key]=[accuracy_stats_task[key]]
            performance_stats['auprc']=[auprc_task]
            performance_stats['auroc']=[auroc_task]
            performance_stats['recall_at_fdr_50']=[recall[0]]
            performance_stats['recall_at_fdr_20']=[recall[1]]
            performance_stats['recall_at_fdr_10']=[recall[2]]            
        else:
            for key in accuracy_stats_task:
                performance_stats[key].append(accuracy_stats_task[key])
            performance_stats['auprc'].append(auprc_task)
            performance_stats['auroc'].append(auroc_task)
            performance_stats['recall_at_fdr_50'].append(recall[0])
            performance_stats['recall_at_fdr_20'].append(recall[1])
            performance_stats['recall_at_fdr_10'].append(recall[2])
    print(str(performance_stats))
    return performance_stats  


def get_performance_metrics_regression(predictions,true_y):
    print(predictions.shape)
    print(true_y.shape) 
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    performance_stats=None
    for c in range(num_cols):
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task,true_y_for_task)
        spearman_task=spearmanr(predictions_for_task_filtered,true_y_for_task_filtered)
        pearson_task=pearsonr(predictions_for_task_filtered,true_y_for_task_filtered)
        #get correlation for non-zero true values
        non_zero_true=true_y_for_task_filtered[true_y_for_task_filtered!=0]
        non_zero_predicted=predictions_for_task_filtered[true_y_for_task_filtered!=0]
        spearman_task_nonzero=spearmanr(non_zero_predicted,non_zero_true)
        pearson_task_nonzero=pearsonr(non_zero_predicted,non_zero_true)
        
        if performance_stats==None:
            performance_stats={'spearmanr':[spearman_task],
                               'pearsonr':[pearson_task],
                               'spearmanr_nonzerobins':[spearman_task_nonzero],
                               'pearsonr_nonzerobins':[pearson_task_nonzero]}
        else:
            performance_stats['spearmanr'].append(spearman_task)
            performance_stats['pearsonr'].append(pearson_task)
            performance_stats['spearman_nonzerobins'].append(spearman_task_nonzero)
            performance_stats['pearson_nonzerobins'].append(pearson_task_nonzero)            
    return performance_stats  


def get_performance_metrics(

def get_performance_metrics(predictions,args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args)
    labels=predictions['labels']
    model_predictions=predictions['predictions']
    #if calibration has been used, we want accuracy metrics on the calibrated predictions (last entry in predictions list) 
    if ((args.calibrate_classification==True) or (args.calibrate_regression==True)): 
        model_predictions=predictions['calibrated_predictionsn']
    if args.performance_metrics_classification_file is not None:
        print("calculating classification performance metrics...")
        performance_metrics_classification=get_performance_metrics_classification(model_predictions,labels)
        print("writing classification performance metrics to file...") 
        write_performance_metrics(args.performance_metrics_classification_file,performance_metrics_classification,model_predictions.columns) 
    elif args.performance_metrics_regression_file is not None:
        print("calculating regression performance metrics...") 
        performance_metrics_regression=get_performance_metrics_regression(model_predictions,labels)
        print("writing regression performance metrics to file...") 
        write_performance_metrics(args.performance_metrics_regression_file,performance_metrics_regression,model_predictions.columns)
    elif args.performance_metrics_profile_file is not None:
        print("calculating profile model performance metrics")
        coords=predictions['coords']
        performance_metrics_profile=get_performance_metrics_profile(model_predictions,
                                                                    labels,
                                                                    coords,
                                                                    args.profile_model_tasks,
                                                                    args.profile_model_task_indices)
        print("writing profile performance metrics to file...")
        write_performance_metrics(args.performance_metrics_profile_file,performance_metrics_profile,args.profile_models_tasks)
    
#write performance metrics to output file: 
def write_performance_metrics(output_file,metrics_dict,tasks):
    metrics_df=pd.DataFrame(metrics_dict,index=tasks).transpose()
    metrics_df.to_csv(output_file,sep='\t',header=True,index=True)
    print(metrics_df)

def main():
    args=parse_args()
    with open(args.predictions_pickle_to_load,'rb') as handle:
        predictions=pickle.load(handle)
    get_performance_metrics(predictions, args)
    
if __name__=="__main__":
    main()
    
