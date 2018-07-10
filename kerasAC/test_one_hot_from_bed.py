import pysam
import pybedtools
import pandas as pd
import numpy as np 
import pdb

ref="/srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa"
fasta_ref = pysam.FastaFile(ref)
train_path="/srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/gecco_v2.train.bed"
#load the train data as a pandas dataframe, skip the header
data=pd.read_csv(train_path,header=0,sep='\t',index_col=[0,1,2])
ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
#iterate through batches and one-hot-encode on the fly
start_index=75000
num_generated=77009
total_entries=data.shape[0]
batch_size=1000
while True:
        if(num_generated >=total_entries):
                start_index=0
        end_index=start_index+batch_size
        #get seq positions
        bed_entries=[(data.index[i]) for i in range(start_index,end_index)]
        #get sequences
        seqs=[fasta_ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        seqs=np.array([[ltrdict[x] for x in seq] for seq in seqs])
        #expand dimension of 1
        x_batch=np.expand_dims(seqs,1)
        y_batch=data[start_index:end_index]
        num_generated+=batch_size
        start_index=end_index
        print('done!')






