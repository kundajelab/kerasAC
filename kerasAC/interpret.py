#compute gradient x input for tensorflow models.
import argparse
import pdb
import pickle
from deeplift.conversion import kerasapi_conversion as kc
import keras.backend as K
import numpy as np
import pyBigWig
from .config import args_object_from_args_dict
from .generators import *
from .predict import get_model

def parse_args():
    parser=argparse.ArgumentParser(description="get gradient x input for a model",add_help=True)
    parser.add_argument("--model_hdf5")
    parser.add_argument("--w0",nargs="+",type=float,default=None)
    parser.add_argument("--w1",nargs="+",type=float,default=None)
    parser.add_argument("--w1_w0_file",default=None)
    parser.add_argument("--predictions_pickle")
    parser.add_argument("--interpret_chroms",nargs="*") 
    parser.add_argument("--interpretation_outf",default=None)
    parser.add_argument("--flank",default=None,type=int)
    parser.add_argument("--method",choices=['gradxinput','deeplift'],default="deeplift")
    parser.add_argument('--batch_size',type=int,help='batch size to use to make model predictions',default=50)
    parser.add_argument('--ref_fasta',default="/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa")
    parser.add_argument('--background_freqs',default=None)
    parser.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file or hammock) and expanded args.flank to the left and right")
    parser.add_argument('--task_id',type=int)
    parser.add_argument('--squeeze_input_for_gru',default=False,action='store_true')
    parser.add_argument('--assembly',default='hg19')
    parser.add_argument('--chromsizes',default='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes')
    parser.add_argument("--precision_thresh",type=float,default=0.9,help="threshold for precision that is used in determining the probability cutoff to use in calling positive predictions")
    parser.add_argument("--yaml",default=None)
    parser.add_argument("--json",default=None)
    parser.add_argument("--expand_dims",default=True)
    parser.add_argument("--tasks",nargs="*",default=None)     
    return parser.parse_args()


def get_deeplift_function(args):
    # convert to deeplift model and get scoring function
    deeplift_model = kc.convert_model_from_saved_files(args.model_hdf5,verbose=False)

    #get the deeplift score with respect to the logit 
    score_func = deeplift_model.get_target_contribs_func(
        find_scores_layer_idx=0,
        target_layer_idx=-2)
    return score_func

def get_deeplift_references(args):
    if args.background_freqs==None:
        # use a 40% GC reference
        input_references = [np.array([0.3, 0.2, 0.2, 0.3])[None, None, None, :]]
    else:
        input_references=[np.array(args.background_freqs)[None,None,None,:]]
    return input_references

def add_bigwig_header(bw,assembly):
    if assembly=='hg19':
        bw.addHeader([('chr1',249250621),
                      ('chr2',243199373),                      
                      ('chr3',198022430),
                      ('chr4',191154276),
                      ('chr5',180915260),
                      ('chr6',171115067),
                      ('chr7',159138663),
                      ('chr8',146364022),
                      ('chr9',141213431),
                      ('chr10',135534747),
                      ('chr11',135006516),
                      ('chr12',133851895),
                      ('chr13',115169878),
                      ('chr14',107349540),
                      ('chr15',102531392),
                      ('chr16',90354753),
                      ('chr17',81195210),
                      ('chr18',78077248),
                      ('chr19',59128983),
                      ('chr20',63025520),
                      ('chr21',48129895),
                      ('chr22',51304566),
                      ('chrX',155270560),
                      ('chrY',59373566)])
        return bw
    else:
        raise Exception("implement bigWig header for this assembly!")

def get_chromsizes(f):
    data=open(f,'r').read().strip().split('\n')
    chromsize_dict=dict()
    for line in data:
        tokens=line.split()
        chromsize_dict[tokens[0]]=int(tokens[1])
    return chromsize_dict



def interpret(args):
    print("starting interpretation")
    if type(args)==type({}):
        args=args_object_from_args_dict(args)
    if args.method=="deeplift":
        # get deeplift scores
        score_func=get_deeplift_function(args)
        input_references=get_deeplift_references(args)
    elif args.method=="gradxinput":
        #calculate gradient x input 
        model=get_model(args)
        grad_tensor=K.gradients(model.layers[-2].output,model.layers[0].input)
        grad_func = K.function([model.layers[0].input,K.learning_phase()], grad_tensor)
    else:
        raise Exception("method must be one of 'deeplift' or 'gradxinput'")
    
    data_generator=TruePosGenerator(args.predictions_pickle,
                                    args.ref_fasta,
                                    batch_size=args.batch_size,
                                    precision_thresh=args.precision_thresh,
                                    expand_dims=args.expand_dims,
                                    tasks=args.tasks)
    print("made data generator!") 
    tasks=list(data_generator.columns)
    bigwigs={}
    for task in tasks:
        bw=pyBigWig.open(args.interpretation_outf+"."+task,'w')
        add_bigwig_header(bw,args.assembly)
        bigwigs[task]=bw
    print("initialized output bigwig files") 
    chromsize_dict=get_chromsizes(args.chromsizes)
    
    #iterate through the batches and get interpretation scores for each task
    num_tasks=len(tasks)
    num_batches=len(data_generator)
    for i in range(num_batches):
        print(str(i)+"/"+str(num_batches)) 
        bed_entries,x,y=data_generator[i]
        for t_index in range(num_tasks):
            cur_task=tasks[t_index]
            if args.method=="deeplift":
                # get deeplift scores
                cur_scores = score_func(
                    task_idx=t_index,
                    input_data_list=[x],
                    batch_size=x.shape[0],
                    progress_update=None,
                    input_references_list=input_references)*x
            else:
                gradient = grad_func([x, False])[0]
                normed_gradient = gradient-np.mean(gradient, axis=3)[:,:,:,None]
                cur_scores = normed_gradient*x
            #save to bigwig 
            chroms = [entry[0] for entry in bed_entries]
            start_vals=[entry[1] for entry in bed_entries] 
            for score_index in range(len(cur_scores)):
                base_scores=np.ndarray.tolist(np.sum(cur_scores[score_index].squeeze(),axis=1))
                try:
                    bigwigs[cur_task].addEntries(chroms[score_index],start_vals[score_index],values=base_scores,span=1,step=1)
                except:
                    continue
    for cur_task in tasks:
        bigwigs[cur_task].close()
            
def main():
    args=parse_args()
    interpret(args) 
    
if __name__=="__main__":
    main()
    
