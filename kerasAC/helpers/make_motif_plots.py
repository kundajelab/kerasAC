import argparse
from matplotlib import pyplot as plt 
from modisco.visualization import viz_sequence
import os
import numpy as np

def _plot_weights(array,
                  path,
                  figsize=(10,3),
                 **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    viz_sequence.plot_weights_given_ax(ax=ax, array=array,**kwargs)
    plt.savefig(path)
    plt.close()




def parse_args():
    parser=argparse.ArgumentParser(description="make plots of motif files given a cwm. useful for meme motif plots")
    parser.add_argument("--motif_file_dir")
    parser.add_argument("--out_dir")
    return parser.parse_args()

def main():
    background = np.array([0.25, 0.25, 0.25, 0.25])
    args=parse_args()
    #get all the files in the directory
    for fname in os.listdir(args.motif_file_dir):
        print(fname) 
        #load & transpose the pfm
        ppm=np.transpose(np.loadtxt(os.path.join(args.motif_file_dir,fname)))
        _plot_weights(viz_sequence.ic_scale(ppm, background=background),
                      path=os.path.join(args.out_dir,fname+'.png'))
                      

if __name__=="__main__":
    main()
    
