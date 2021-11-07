import argparse
import math
import pysam 
import shap 
import tensorflow
from deeplift.dinuc_shuffle import dinuc_shuffle
from scipy.spatial.distance import jensenshannon
from scipy.special import logit, softmax


tensorflow.compat.v1.disable_v2_behavior()
import kerasAC
import matplotlib
import pandas as pd
from kerasAC.interpret.deepshap import *
from kerasAC.interpret.profile_shap import *
from kerasAC.util import * 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from kerasAC.custom_losses import *
from kerasAC.metrics import *

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for SNP scoring")
    parser.add_argument("--model_hdf5")
    parser.add_argument("--peak_file")
    parser.add_argument("--npeaks_to_sample",type=int,default=30000)
    parser.add_argument("--out_prefix")
    parser.add_argument(
        "--ref_fasta", default="/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    parser.add_argument("--dinuc_shuffle_input",action='store_true',default=False) 
    parser.add_argument("--chrom_sizes", default="/data/hg38.chrom.sizes")
    parser.add_argument("--flank_size", type=int, default=500)
    parser.add_argument("--batch_size",type=int,default=100)
    return parser.parse_args()




def load_model_wrapper(model_hdf5):
    # load the model!
    custom_objects = {
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "f1": f1,
        "ambig_binary_crossentropy": ambig_binary_crossentropy,
        "ambig_mean_absolute_error": ambig_mean_absolute_error,
        "ambig_mean_squared_error": ambig_mean_squared_error,
        "MultichannelMultinomialNLL": MultichannelMultinomialNLL,
    }
    get_custom_objects().update(custom_objects)
    return load_model(model_hdf5)





def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in [0]:
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (
                hypothetical_input[None, :, :] - bg_data[l]
            )
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(
                hypothetical_contribs, axis=-1
            )
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    to_return.append(np.zeros_like(orig_inp[1]))
    return to_return


def shuffle_several_times(s):
    numshuffles = 20
    return [
        np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
        np.array([s[1] for i in range(numshuffles)]),
    ]

def main():
    args = parse_args()
    chrom_sizes=open(args.chrom_sizes,'r').read().strip().split('\n')
    chrom_size_dict={}
    for line in chrom_sizes:
        tokens=line.split('\t')
        chrom_size_dict[tokens[0]]=int(tokens[1])
        
    ref=pysam.FastaFile(args.ref_fasta)
    # load the model
    model = load_model_wrapper(args.model_hdf5)
    print("loaded model") 
    # create the count & profile explainers
    model_wrapper = (model.input, model.outputs[1][:, 0:1])
    count_explainer = shap.DeepExplainer(
        model_wrapper,
        data=create_background_atac,
        combine_mult_and_diffref=combine_mult_and_diffref_atac
    )
    prof_explainer = create_explainer(model, ischip=False, task_index=0)
    print("made explainers") 
    #read in the peaks
    peaks=pd.read_csv(args.peak_file,header=None,sep='\t')
    nrow=peaks.shape[0]
    tosample=round(int(args.npeaks_to_sample)/nrow,2)
    peaks = peaks.sample(frac=tosample).reset_index(drop=True)
    nrow=peaks.shape[0]
    print("sampled peaks:"+str(nrow))

    #allocate space for numpy arrays for modisco 
    hypothetical_profile_scores=np.empty((nrow,2*args.flank_size,4))
    hypothetical_count_scores=np.empty((nrow,2*args.flank_size,4))
    observed_profile_scores=np.empty((nrow,2*args.flank_size,4))
    observed_count_scores=np.empty((nrow,2*args.flank_size,4))
    seq=np.empty((nrow,2*args.flank_size,4))
    print("pre-allocted output arrays")
    
    #generate one-hot-encoded inputs
    start_index=0
    while start_index < nrow:
        cur_batch_size=min(args.batch_size,nrow-start_index)
        print(str(start_index)+":"+str(start_index+cur_batch_size))
        
        batch_chroms=peaks[0][start_index:start_index+cur_batch_size].tolist() 
        batch_start_pos=peaks[1]+peaks[9]-args.flank_size
        batch_start_pos=batch_start_pos.tolist()
        batch_start_pos=[max(0,i) for i in batch_start_pos]
        batch_start_pos=[min(batch_start_pos[i],chrom_size_dict[batch_chroms[i]]-2*args.flank_size) for i in range(cur_batch_size)]
        seq_batch=[ref.fetch(batch_chroms[i],batch_start_pos[i],batch_start_pos[i]+2*args.flank_size) for i in range(cur_batch_size)]
        if args.dinuc_shuffle_input is True:
            seq_batch=[dinuc_shuffle(i) for i in seq_batch]
        seq_batch=one_hot_encode(seq_batch)
            

        seq[start_index:start_index+cur_batch_size,:,:]=seq_batch
        #get the hypothetical scores for the batch
        hypothetical_profile_scores[start_index:start_index+cur_batch_size,:,:]= prof_explainer(seq_batch, None)
        observed_profile_scores[start_index:start_index+cur_batch_size,:,:]=hypothetical_profile_scores[start_index:start_index+cur_batch_size,:,:]*seq_batch
        hypothetical_count_scores[start_index:start_index+cur_batch_size,:,:]= np.squeeze(count_explainer.shap_values(seq_batch)[0])
        observed_count_scores[start_index:start_index+cur_batch_size,:,:]=hypothetical_count_scores[start_index:start_index+cur_batch_size,:,:]*seq_batch
        start_index+=args.batch_size
    #save
    print("saving outputs") 
    np.save(args.out_prefix+'.hyp.profile.npy',hypothetical_profile_scores)
    np.save(args.out_prefix+'.observed.profile.npy',observed_profile_scores)
    np.save(args.out_prefix+'.hyp.count.npy',hypothetical_count_scores)
    np.save(args.out_prefix+'.observed.count.npy',observed_count_scores)
    np.save(args.out_prefix+'.seq.npy',seq)


if __name__ == "__main__":
    main()
