from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import warnings
import numpy as np
import pysam
import pandas as pd
from scipy.stats import spearmanr, pearsonr 
from collections import OrderedDict, defaultdict
from .utils import * 

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

    
