from .splits import *
from .config import args_object_from_args_dict
from .train import *
from .predict import *
from .interpret import * 
import argparse
import pdb

def parse_args():
    parser=argparse.ArgumentParser(add_help=True)
    parser.add_argument("--multi_gpu",action="store_true",default=False)
    parser.add_argument("--assembly",default="hg19")
    parser.add_argument("--data_path",help="path that stores training/validation/test data")
    parser.add_argument("--model_hdf5",required=True)
    parser.add_argument("--batch_size",type=int,default=1000)
    parser.add_argument("--init_weights",default=None)
    parser.add_argument("--ref_fasta",default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
    parser.add_argument("--w1_w0_file",default=None)
    parser.add_argument("--save_w1_w0", default=None,help="output text file to save w1 and w0 to")
    parser.add_argument("--weighted",action="store_true")
    parser.add_argument('--w1',nargs="*", type=float, default=None)
    parser.add_argument('--w0',nargs="*", type=float, default=None)

    parser.add_argument("--from_checkpoint_weights",default=None)
    parser.add_argument("--from_checkpoint_arch",default=None)
    parser.add_argument("--num_tasks",required=True,type=int)
    parser.add_argument("--num_train",type=int,default=700000)
    parser.add_argument("--num_valid",type=int,default=150000)

    #add functionality to train on individuals' allele frequencies
    parser.add_argument("--vcf_file",default=None)
    parser.add_argument("--global_vcf",action="store_true")
    parser.add_argument("--revcomp",action="store_true")
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--patience",type=int,default=3)
    parser.add_argument("--patience_lr",type=int,default=2,help="number of epochs with no drop in validation loss after which to reduce lr")
    parser.add_argument("--architecture_spec",type=str,default="basset_architecture_multitask")
    parser.add_argument("--architecture_from_file",type=str,default=None)
    parser.add_argument("--tensorboard",action="store_true")
    parser.add_argument("--tensorboard_logdir",default="logs")
    parser.add_argument("--squeeze_input_for_gru",action="store_true")
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--train_upsample", type=float, default=None)
    parser.add_argument("--valid_upsample", type=float, default=None)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--max_queue_size",type=int,default=100)
    parser.add_argument('--weights',help='weights file for the model')
    parser.add_argument('--yaml',help='yaml file for the model')
    parser.add_argument('--json',help='json file for the model')
    parser.add_argument('--predict_chroms',default=None) 
    parser.add_argument('--data_hammock',help='input file is in hammock format, with unique id for each peak')
    parser.add_argument('--variant_bed')
    parser.add_argument('--predictions_pickle',help='name of pickle to save predictions',default=None)
    parser.add_argument('--performance_metrics_classification_file',help='file name to save classification performance metrics',default=None)
    parser.add_argument('--performance_metrics_regression_file',help='file name to save regression performance metrics',default=None)
    parser.add_argument('--predictions_pickle_to_load',help="if predictions have already been generated, provide a pickle with them to just compute the performance metrics",default=None)
    parser.add_argument('--background_freqs',default=None)
    parser.add_argument('--flank',default=500,type=int)
    parser.add_argument('--mask',default=10,type=int)
    parser.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    parser.add_argument("--interpret_chroms",nargs="*") 
    parser.add_argument("--interpretation_outf",default=None)
    parser.add_argument("--method",choices=['gradxinput','deeplift'],default="deeplift")
    parser.add_argument('--task_id',type=int)
    parser.add_argument('--chromsizes',default='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes')
    parser.add_argument("--interpret",action="store_true",default=False)
    parser.add_argument("--calibrate_classification",action="store_true",default=False)
    parser.add_argument("--calibrate_regression",action="store_true",default=False) 

    return parser.parse_args()


def cross_validate(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 

    #run training on each of the splits
    if args.assembly not in splits:
        raise Exception("Unsupported genome assembly:"+args.assembly+". Supported assemblies include:"+str(splits.keys())+"; add splits for this assembly to splits.py file")
    args_dict=vars(args)
    print(args_dict) 
    base_model_file=args_dict['model_hdf5']
    base_performance_classification_file=args_dict['performance_metrics_classification_file']
    base_performance_regression_file=args_dict['performance_metrics_regression_file']
    base_interpretation=args_dict['interpretation_outf']
    base_predictions_pickle=args_dict['predictions_pickle']


    for split in splits[args.assembly]:
        print("Starting split:"+str(split))
        test_chroms=splits[args.assembly][split]['test']
        validation_chroms=splits[args.assembly][split]['valid']
        train_chroms=list(set(chroms[args.assembly])-set(test_chroms+validation_chroms))

        #convert args to dict
        args_dict=vars(args)
        args_dict['train_chroms']=train_chroms
        args_dict['validation_chroms']=validation_chroms
        
           
        #set the training arguments specific to this fold 
        args_dict['model_hdf5']=base_model_file+"."+str(split)
        print("Training model") 
        train(args_dict)
        
        #set the prediction arguments specific to this fold
        if args.save_w1_w0!=None:
            args_dict["w1_w0_file"]=args.save_w1_w0
        if base_performance_classification_file!=None:
            args_dict['performance_metrics_classification_file']=base_performance_classification_file+"."+str(split)
        if base_performance_regression_file!=None:
            args_dict['performance_metrics_regression_file']=base_performance_regression_file+"."+str(split)
        if base_predictions_pickle!=None:
            args_dict['predictions_pickle']=base_predictions_pickle+"."+str(split) 
        args_dict['predict_chroms']=test_chroms
        print("Calculating predictions on the test fold") 
        predict(args_dict)

        if args.interpret==True:
            args_dict['interpret_chroms']=test_chroms
            args_dict['interpretation_outf']=base_interpretation+'.'+str(split)
            print("Running interpretation on the test fold") 
            interpret(args_dict)
        
        
def main():
    args=parse_args()
    cross_validate(args) 

if __name__=="__main__":
    main()
