import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt 
import viz_sequence

def parse_args():
    parser=argparse.ArgumentParser(description="trim modisco hits to regions with high signal")
    parser.add_argument("--modisco_hits")
    parser.add_argument("--trim_thresh",type=float,default=0.45)
    parser.add_argument("--trim_extend",type=int,default=1)    
    return parser.parse_args()



def _plot_weights(array,
                  path,
                  figsize=(10,3),
                  **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax=ax, array=array,**kwargs)
    plt.savefig(path)
    plt.close()
    
def main():
    args=parse_args()
    #load the modisco hdf5 file
    hdf5_results=h5py.File(args.modisco_hits,'r')

    for metacluster_name in hdf5_results["metacluster_idx_to_submetacluster_results"]:
        metacluster = hdf5_results["metacluster_idx_to_submetacluster_results"][metacluster_name]
        if metacluster['activity_pattern'][0] == 1:
            all_pattern_names = [x.decode("utf-8") for x in list(metacluster["seqlets_to_patterns_result"]["patterns"]["all_pattern_names"][:])]
            for pattern_name in all_pattern_names:
                cwm_fwd = np.array(metacluster['seqlets_to_patterns_result']['patterns'][pattern_name]['task0_contrib_scores']['fwd'])
                cwm_rev = np.array(metacluster['seqlets_to_patterns_result']['patterns'][pattern_name]['task0_contrib_scores']['rev'])
                
                score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
                score_rev = np.sum(np.abs(cwm_rev), axis=1)

                trim_thresh_fwd = np.max(score_fwd) * args.trim_thresh
                trim_thresh_rev = np.max(score_rev) * args.trim_thresh
                
                pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]
                
                start_fwd, end_fwd = max(np.min(pass_inds_fwd) - args.trim_extend, 0), min(np.max(pass_inds_fwd) + args.trim_extend + 1, len(score_fwd) + 1)
                start_rev, end_rev = max(np.min(pass_inds_rev) - args.trim_extend, 0), min(np.max(pass_inds_rev) + args.trim_extend + 1, len(score_rev) + 1)

                trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
                trimmed_cwm_rev = cwm_rev[start_rev:end_rev]
                
                _plot_weights(trimmed_cwm_fwd,
                              path='.'.join([metacluster_name,pattern_name,'cwm.fwd.png']))
                _plot_weights(trimmed_cwm_rev,
                              path='.'.join([metacluster_name,pattern_name,'cwm.fwd.png']))
                

if __name__=="__main__":
    main()
    
