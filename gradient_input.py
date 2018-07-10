#compute gradient x input for tensorflow models.
import argparse
from get_model_predictions import * 
def parse_args():
    parser=argparse.ArgumentParser(description="get gradient x input for a model")
    parser.add_argument("--model_hdf5")
    parser.add_argument("--w0",nargs="+",type=float)
    parser.add_argument("--w1",nargs="+",type=float)
    parser.add_argument("--input_bed")
    parser.add_argument("--outf")
    parser.add_argument("--ref")
    parser.add_argument("--flank",default=500,type=int)
    return parser.parse_args()
def get_predictions_bed(args,model):
    data=[i.split('\t') for i in open(args.input_bed,'r').read().strip().split('\n')]
    #original
    import pysam
    ref=pysam.FastaFile(args.ref)
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    seqs=[]
    for entry in data:
        #introduce the specified variant 
        start_pos=int(entry[1])-(args.flank)
        end_pos=int(entry[1])+args.flank
        seq=ref.fetch(entry[0],start_pos,end_pos)
        seqs.append(seq)
    seqs=np.array([[ltrdict[x] for x in seq] for seq in seqs])    

    seqs=np.expand_dims(seqs,1)    
    predictions=model.predict(seqs)
    return predictions,seqs


def main():
    args=parse_args()
    model=get_model(args)
    predictions,inputs=get_predictions_bed(args,model)
    import tensorflow as tf
    grad_tensor=K.gradients(model.layers[-2].output,model.layers[0].input)
    grad_func = K.function([model.layers[0].input,K.learning_phase()], grad_tensor)
    gradient = grad_func([inputs, False])[0]
    normed_gradient = gradient-np.mean(gradient, axis=3)[:,:,:,None]
    normed_grad_times_inp = normed_gradient*inputs

    #plot the pssm
    from deeplift.visualization import viz_sequence 
    viz_sequence.plot_weights(scores_for_idx, subticks_frequency=10, highlight=highlight)
    
if __name__=="__main__":
    main()
    
