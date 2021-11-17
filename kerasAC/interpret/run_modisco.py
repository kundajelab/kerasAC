import h5py
import numpy as np
import modisco
import sys

import argparse
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--hyp_imp")
    parser.add_argument("--imp")
    parser.add_argument("--onehot")
    parser.add_argument("--outfile")
    parser.add_argument("--cores",type=int)
    parser.add_argument("--seqlets",type=int)
    return parser.parse_args()

def main():
    args=parse_args() 
    hyp_impscores = np.load(args.hyp_imp)
    impscores = np.load(args.imp)
    onehot_seqs = np.load(args.onehot)
    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=20000)

    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        max_seqlets_per_metacluster=args.seqlets,
        sliding_window_size=21,
        flank_size=10,
        target_seqlet_fdr=0.05,
        min_passing_windows_frac=0.03,
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            n_cores=args.cores,
            trim_to_window_size=30,
            initial_flank_to_add=10,
            final_min_cluster_size=30))(task_names=["task0"],
                                        contrib_scores={'task0': impscores},
                                        hypothetical_contribs={'task0': hyp_impscores},
                                        null_per_pos_scores=null_per_pos_scores,
                                        one_hot=onehot_seqs)
    
    h5f = h5py.File(args.outfile, 'w')
    tfmodisco_results.save_hdf5(h5f)
    h5f.close()


if __name__=='__main__':
    main()

