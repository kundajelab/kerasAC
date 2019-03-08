import argparse
import pdb 
def args_object_from_args_dict(args_dict):
    #create an argparse.Namespace from the dictionary of inputs
    args_object=argparse.Namespace()
    #set the defaults for training/prediction/interpretation/cross-validation
    #training
    vars(args_object)['multigpu']=False
    vars(args_object)['batch_size']=1000
    vars(args_object)['num_train']=700000
    vars(args_object)['num_valid']=150000
    vars(args_object)['train_chroms']=None
    vars(args_object)['validation_chroms']=None 
    vars(args_object)['ref_fasta']='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa'
    vars(args_object)['init_weights']=None
    vars(args_object)['w1_w0_file']=None
    vars(args_object)['squeeze_input_for_gru']=False 
    vars(args_object)['from_checkpoint_weights']=None
    vars(args_object)['from_checkpoint_arch']=None
    vars(args_object)['vcf_file']=None
    vars(args_object)['epochs']=40
    vars(args_object)['patience']=3
    vars(args_object)['patience_lr']=2
    vars(args_object)['architecture_spec']="basset_architecture_multitask"
    vars(args_object)['architecture_from_file']=None
    vars(args_object)['tensorborad_logdir']='logs'
    vars(args_object)['seed']=1234
    vars(args_object)['train_path']=None
    vars(args_object)['valid_path']=None 
    vars(args_object)['train_upsample']=None
    vars(args_object)['valid_upsample']=None
    vars(args_object)['threads']=1
    vars(args_object)['max_queue_size']=100
    vars(args_object)['save_weights']=None
    vars(args_object)['w1']=None
    vars(args_object)['w0']=None
    vars(args_object)['yaml']=None
    vars(args_object)['json']=None
    
    
    #prediction
    vars(args_object)['predict_chroms']=None
    vars(args_object)['prediction_pickle']=None
    vars(args_object)['performance_metrics_classification_file']=None
    vars(args_object)['performance_metrics_regression_file']=None
    vars(args_object)['predictions_pickle_to_load']=None
    vars(args_object)['sequential']=False
    vars(args_object)['background_freqs']=None
    vars(args_object)['flank']=500
    vars(args_object)['mask']=10
    vars(args_object)['center_on_summit']=False
    vars(args_object)['calibrate_classification']=False
    vars(args_object)['calibrate_regression']=False     

    #cross-validation
    vars(args_object)['assembly']='hg19'

    #interpret
    vars(args_object)['method']='deeplift'
    vars(args_object)['deeplift_ref']='shuffled_ref' 
    vars(args_object)['background_freqs']=None
    vars(args_object)['chromsizes']="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes"
    vars(args_object)['precision_thresh']=0.90
    for key in args_dict:
        vars(args_object)[key]=args_dict[key]
    args=args_object
    return args

    
    
