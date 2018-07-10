from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import kerasAC.util 
from collections import OrderedDict, defaultdict
import sys 
import pdb

def remove_ambiguous_peaks(predictions, true_y): 
    indices_to_remove = np.where(true_y < 0)
    true_y_filtered = np.delete(true_y, indices_to_remove)
    predictions_filtered = np.delete(predictions, indices_to_remove)
    return predictions_filtered, true_y_filtered


def auroc_func(predictions, true_y,thresh=None):
    [num_rows, num_cols] = true_y.shape 
    aurocs=[]
    for c in range(num_cols): 
        true_y_for_task = true_y[:,c]
        predictions_for_task = predictions[:,c]
        predictions_for_task_filtered, true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        try:
            task_auroc = roc_auc_score(y_true=true_y_for_task_filtered,
                                   y_score=predictions_for_task_filtered)
        except Exception as e:
            #if there is only one class in the batch of true_y, then auROC cannot be calculated
            print("Could not calculate auROC:")
            print(str(e))
            task_auroc=None 
        aurocs.append(task_auroc) 
    return aurocs

def auprc_func(predictions, true_y,thresh=None):
    # sklearn only supports 2 classes (0,1) for the auPRC calculation
    [num_rows, num_cols]=true_y.shape 
    auprcs=[]
    for c in range(num_cols): 
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = \
         remove_ambiguous_peaks(predictions_for_task, true_y_for_task)
        try:
            task_auprc = average_precision_score(true_y_for_task_filtered, predictions_for_task_filtered);
        except:
            print("Could not calculated auPRC:")
            print(sys.exc_info()[0])
            task_auprc=None 
        auprcs.append(task_auprc) 
    return auprcs;

def get_accuracy_stats_for_task(predictions, true_y, c):
    true_y_for_task=np.squeeze(true_y[:,c])
    predictions_for_task=np.squeeze(predictions[:,c])
    predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task,true_y_for_task)
    predictions_for_task_filtered_round = np.array([round(el) for el in predictions_for_task_filtered])
    accuratePredictions = predictions_for_task_filtered_round==true_y_for_task_filtered;

    numPositives_forTask=np.sum(true_y_for_task_filtered==1,axis=0,dtype="float");
    numNegatives_forTask=np.sum(true_y_for_task_filtered==0,axis=0,dtype="float"); 

    accuratePredictions_positives = np.sum(accuratePredictions*(true_y_for_task_filtered==1),axis=0);
    accuratePredictions_negatives = np.sum(accuratePredictions*(true_y_for_task_filtered==0),axis=0);
    returnDict = {
        'accuratePredictions': accuratePredictions,
        'numPositives_forTask': numPositives_forTask,
        'numNegatives_forTask': numNegatives_forTask,
        'true_y_for_task_filtered': true_y_for_task_filtered,
        'predictions_for_task_filtered': predictions_for_task_filtered,
        'accuratePredictions_positives': accuratePredictions_positives,
        'accuratePredictions_negatives': accuratePredictions_negatives
    }
    return returnDict


def unbalanced_accuracy(predictions, true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    unbalanced_accuracies = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        unbalancedAccuracy_forTask = (r['accuratePredictions_positives'] + r['accuratePredictions_negatives'])/(r['numPositives_forTask']+r['numNegatives_forTask']).astype("float");
        unbalanced_accuracies.append(unbalancedAccuracy_forTask) 
    return unbalanced_accuracies;

def positives_accuracy(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    positive_accuracies = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        positiveAccuracy_forTask = float(r['accuratePredictions_positives'])/float(r['numPositives_forTask'])
        positive_accuracies.append(positiveAccuracy_forTask) 
    return positive_accuracies;

def negatives_accuracy(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    negative_accuracies = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        negativeAccuracy_forTask = float(r['accuratePredictions_negatives'])/float(r['numNegatives_forTask'])
        negative_accuracies.append(negativeAccuracy_forTask) 
    return negative_accuracies;

def num_positives(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    num_positives = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        num_positives.append(r['numPositives_forTask']) 
    return num_positives;

def num_negatives(predictions,true_y,thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    num_negatives = []
    for c in range(num_cols):
        r = get_accuracy_stats_for_task(predictions, true_y, c)
        num_negatives.append(r['numNegatives_forTask'])
    return num_negatives;

def balanced_accuracy(predictions, true_y, thresh=None):
    assert predictions.shape==true_y.shape;
    assert len(predictions.shape)==2;
    [num_rows, num_cols]=true_y.shape 
    balanced_accuracies = []
    for c in range(num_cols): 
        r = get_accuracy_stats_for_task(predictions, true_y, c)
    
        positivesAccuracy_forTask = r['accuratePredictions_positives']/r['numPositives_forTask'];
        negativesAccuracy_forTask = r['accuratePredictions_negatives']/r['numNegatives_forTask'];

        balancedAccuracy_forTask= (positivesAccuracy_forTask+negativesAccuracy_forTask)/2;
        balanced_accuracies.append(balancedAccuracy_forTask) 
    return balanced_accuracies
def recall_at_fdr_function(predictions,true_y,thresh):
    if float(thresh)>1: 
        thresh=float(thresh)/100 
    [num_rows, num_cols]=true_y.shape
    recall_at_fdr_vals=[]
    for c in range(num_cols): 
        true_y_for_task=np.squeeze(true_y[:,c])
        predictions_for_task=np.squeeze(predictions[:,c])
        predictions_for_task_filtered,true_y_for_task_filtered = remove_ambiguous_peaks(predictions_for_task, true_y_for_task)

        #group by predicted prob
        predictedProbToLabels = defaultdict(list)
        for predictedProb, trueY in zip(predictions_for_task_filtered, true_y_for_task_filtered):
            predictedProbToLabels[predictedProb].append(trueY)
        #sort in ascending order of confidence
        sortedThresholds = sorted(predictedProbToLabels.keys())
        toReturnDict = OrderedDict();
        thresholdPairs=[("recallAtFDR"+str(thresh),thresh)]
        #sort desired recall thresholds by descending order of fdr        
        totalPositives = np.sum(true_y_for_task_filtered)
        totalNegatives = np.sum(1-true_y_for_task_filtered)
        #start at 100% recall
        confusionMatrixStatsSoFar = [[0,totalNegatives]
                                    ,[0,totalPositives]]
        recallsForThresholds = []; #for debugging
        fdrsForThresholds = [];

        for threshold in sortedThresholds:
            labelsAtThreshold=predictedProbToLabels[threshold];
            positivesAtThreshold=sum(labelsAtThreshold)
            negativesAtThreshold = len(labelsAtThreshold)-positivesAtThreshold
            
            #when you cross this threshold they all get predicted as negatives.
            confusionMatrixStatsSoFar[0][0] += negativesAtThreshold
            confusionMatrixStatsSoFar[0][1] -= negativesAtThreshold
            confusionMatrixStatsSoFar[1][0] += positivesAtThreshold
            confusionMatrixStatsSoFar[1][1] -= positivesAtThreshold
            totalPredictedPositives = confusionMatrixStatsSoFar[0][1]\
                                  + confusionMatrixStatsSoFar[1][1]
            fdr = 1 - (confusionMatrixStatsSoFar[1][1]/
                       float(totalPredictedPositives))\
                       if totalPredictedPositives > 0 else 0.0
            recall = confusionMatrixStatsSoFar[1][1]/float(totalPositives)
            recallsForThresholds.append(recall)
            fdrsForThresholds.append(fdr)
            #first index of a thresholdPair is the name, second idx
            #is the actual threshold
            while (len(thresholdPairs)>0 and fdr<=thresholdPairs[0][1]):
                toReturnDict[thresholdPairs[0][0]]=recall
                thresholdPairs=thresholdPairs[1::]
            if len(thresholdPairs)==0:
                break;
        for thresholdPair in thresholdPairs:
            toReturnDict[thresholdPairs[0][0]]=0.0
        recall_at_fdr_vals.append(toReturnDict['recallAtFDR'+str(thresh)])
    return recall_at_fdr_vals


AccuracyStats = kerasAC.util.enum(
    auROC="auROC",
    auPRC="auPRC",
    balanced_accuracy="balanced_accuracy",
    unbalanced_accuracy="unbalanced_accuracy",
    positives_accuracy="positives_accuracy",
    negatives_accuracy="negatives_accuracy",
    num_positives="num_positives",
    num_negatives="num_negatives",
    onehot_rows_crossent="onehot_rows_crossent",
    recall_at_fdr50="recallAtFDR50",
    recall_at_fdr20="recallAtFDR20",
    recall_at_fdr10="recallAtFDR10",
    recall_at_fdr1="recallAtFDR1",
    recall_at_fdr="recallAtFDR")

compute_func_lookup = {
    AccuracyStats.auROC: auroc_func,
    AccuracyStats.auPRC: auprc_func,
    AccuracyStats.balanced_accuracy: balanced_accuracy,
    AccuracyStats.unbalanced_accuracy: unbalanced_accuracy,
    AccuracyStats.recall_at_fdr50: recall_at_fdr_function,
    AccuracyStats.recall_at_fdr20: recall_at_fdr_function,
    AccuracyStats.recall_at_fdr10: recall_at_fdr_function,
    AccuracyStats.recall_at_fdr1: recall_at_fdr_function,
    AccuracyStats.recall_at_fdr: recall_at_fdr_function,
    AccuracyStats.positives_accuracy:positives_accuracy,
    AccuracyStats.negatives_accuracy:negatives_accuracy,
    AccuracyStats.num_positives:num_positives,
    AccuracyStats.num_negatives:num_negatives
}
is_larger_better_lookup = {
    AccuracyStats.auROC: True,
    AccuracyStats.auPRC: True,
    AccuracyStats.balanced_accuracy: True,
    AccuracyStats.unbalanced_accuracy: True,
    AccuracyStats.recall_at_fdr50: True,
    AccuracyStats.recall_at_fdr20: True,
    AccuracyStats.recall_at_fdr10: True,
    AccuracyStats.recall_at_fdr1: True,
    AccuracyStats.recall_at_fdr: True
}

multi_level_metrics=["recallAtFDR"]

