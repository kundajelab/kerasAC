import pandas as pd
#Performance metrics for profile models

def parse_args():
    parser=argparse.ArgumentParser(description="performance metrics for profile models")
    parser.add_argument("--genomewide_labels",nargs="+",help="each label hdf5 contains labels for a given task in the model")
    parser.add_argument("--genomewide_predictions",nargs="+",help="each predictin hdf5 contains predictions for a given task from the model")
    parser.add_argument("--summit_labels",nargs="+",help="each label hdf5 contains labels for a given task in the model")
    parser.add_argument("--summit_predictions",nargs="+",help="each predictin hdf5 contains predictions for a given task from the model")
    parser.add_argument("--peak_labels",nargs="+",help="each label hdf5 contains labels for a given task in the model")
    parser.add_argument("--peak_predictions",nargs="+",help="each predictin hdf5 contains predictions for a given task from the model")
    parser.add_argument("--task_names",nargs="+")
    parser.add_argument("--outf")
    parser.add_argument("--bigwig_reps",nargs="+",help="bigwig replicates for calculating upper bound of performance")
    
def mse():
    pass

def pearson():
    pass

def spearman():
    pass

def shuffled_mse_lower_bound():
    pass

def rep_concordance():
    pass

def all_metrics_for_one_label_set(labels_hdf5,predictions_hdf5,rep1_bigwig,rep2_bigwig):
    pass

def all_metrics(args):
    if type(args)==type({}):
        args=argparse.Namespace()
    


def main():
    args=parse_args()
    all_metrics(args) 
    
if __name__=="__main__":
    main()
    



    
