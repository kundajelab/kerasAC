import pandas as pd
import argparse
#from .utils import *
from scipy.stats import spearmanr, pearsonr
from scipy import nanmean, nanstd
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
import matplotlib 
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"]=10,5
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
#Performance metrics for profile models

def parse_args():
    parser=argparse.ArgumentParser(description="performance metrics for profile models")
    parser.add_argument("--labels",help="each label hdf5 contains labels for a given task in the model")
    parser.add_argument("--predictions",help="each predictin hdf5 contains predictions for a given task from the model")
    parser.add_argument("--losses",nargs="+",help="counts or profile")
    parser.add_argument("--loss_suffixes",nargs="+")
    parser.add_argument("--outf")
    parser.add_argument("--title") 
    parser.add_argument("--bigwig_reps",nargs="+",default=None,help="bigwig replicates for calculating upper bound of performance")
    return parser.parse_args() 

def counts_metrics(labels,preds,outf,title):            
    spearman_cor=spearmanr(labels[0].values,preds[0].values)[0]
    pearson_cor=pearsonr(labels[0].values,preds[0].values)[0]
    plt.rcParams["figure.figsize"]=8,8
    plt.figure()
    plt.scatter(labels[0].values, preds[0].values ,alpha=0.1)
    plt.xlabel('Log Count Labels')
    plt.ylabel('Log Count Predictions')
    plt.title("counts:"+title+" spearman R="+str(round(spearman_cor,3))+", Pearson R="+str(round(pearson_cor,3)))
    plt.legend(loc='best')
    plt.xlim(0,11)
    plt.ylim(0,11)
    plt.savefig(outf+".counts.png",format='png',dpi=300)
    return spearman_cor, pearson_cor


def profile_metrics(profile_labels,profile_preds,counts_labels,counts_preds,outf_prefix,title):
    #profile-preds is in logit space
    #get the softmax to put in probability space
    profile_preds_softmax=softmax(profile_preds,axis=1)
    #put the counts in probability space to use jsd
    num_regions=profile_labels.shape[0]
    region_jsd=[]
    outf=open(outf_prefix+".jsd.txt",'w')
    outf.write('Region\tJSD\n')
    for i in range(num_regions): 
        cur_profile_labels_prob=profile_labels.iloc[i].values/sum(profile_labels.iloc[i].values)
        cur_profile_preds_softmax=profile_preds_softmax.iloc[i]
        cur_index=profile_labels.index[i]
        cur_jsd=jensenshannon(cur_profile_labels_prob,cur_profile_preds_softmax)
        region_jsd.append(cur_jsd) 
        outf.write(str(cur_index)+'\t'+str(cur_jsd)+'\n')
    outf.close() 
    #plot jsd histogram
    num_bins=100
    plt.rcParams["figure.figsize"]=8,8
    plt.figure()
    n,bins,patches=plt.hist(region_jsd,num_bins,facecolor='blue',alpha=0.5)
    plt.xlabel('Jensen Shannon Distance Profile Labels and Preds in Probability Space')
    plt.title("profile:"+title)
    plt.savefig(outf_prefix+".jsd.png",format='png',dpi=300)
    #get mean and std
    return nanmean(region_jsd), nanstd(region_jsd)
    
def get_performance_metrics_profile_wrapper(args):
    if type(args)==type({}):
        args=config.args_object_from_args_dict(args)
    labels_and_preds={} 
    for loss_index in range(len(args.losses)):
        cur_loss=args.losses[loss_index]
        cur_loss_suffix=args.loss_suffixes[loss_index]
        cur_pred=pd.read_hdf(args.predictions+"."+cur_loss_suffix,header=None,sep='\t')
        cur_labels=pd.read_hdf(args.labels+"."+cur_loss_suffix,header=None,sp='\t')
        labels_and_preds[cur_loss]={}
        labels_and_preds[cur_loss]['labels']=cur_labels
        labels_and_preds[cur_loss]['predictions']=cur_pred 
    print("loaded labels and predictions")
    spearman_cor,pearson_cor=counts_metrics(labels_and_preds['counts']['labels'],labels_and_preds['counts']['predictions'],args.outf,args.title)
    mean_jsd,std_jsd=profile_metrics(labels_and_preds['profile']['labels'],labels_and_preds['profile']['predictions'],labels_and_preds['counts']['labels'],labels_and_preds['counts']['predictions'],args.outf,args.title)
    outf=open(args.outf+".summary.txt",'w')
    outf.write('Title\tPearson\tSpearman\tMeanJSD\tStdJSD\n')
    outf.write(args.title+'\t'+str(pearson_cor)+'\t'+str(spearman_cor)+'\t'+str(mean_jsd)+'\t'+str(std_jsd)+'\n')
    outf.close() 

def main():
    args=parse_args()
    get_performance_metrics_profile_wrapper(args)
    
    
if __name__=="__main__":
    main()
    



    
