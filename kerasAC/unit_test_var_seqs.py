background_freqs=[0.277,0.277,0.223,0.223]
ref="/srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa"
mask=10
flank=500
variant_bed="/srv/scratch/annashch/gecco/variants.bed"
###
import pysam
import pdb
import random
import numpy as np 
ref=pysam.FastaFile(ref)
data=[i.split('\t') for i in open(variant_bed,'r').read().strip().split('\n')]
ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
#original
seqs=[]
#snp
seqs_snp=[]
#n bases around snp (i.e. knock out an enhancer)
seqs_enhancer=[] 
for entry in data[1::]:
    #introduce the specified variant 
    start_pos=int(entry[1])-(flank)
    end_pos=int(entry[1])+flank
    seq=ref.fetch(entry[0],start_pos,end_pos)
    seqs.append(seq)
    alt_allele=entry[3]
    if alt_allele=="NA":
        #randomly insert another base
        ref_allele=seq[flank-1]
        options=['A','C','G','T']
        options.remove(ref_allele)
        alt_allele=options[random.randint(0,2)]
    seq=seq[0:flank-1]+alt_allele+seq[flank:len(seq)]
    seqs_snp.append(seq)
    seq=np.array([ltrdict[x] for x in seq])
    start_mask=flank-1-mask
    end_mask=flank-1+mask
    seq=seq*1.0
    seq[start_mask:end_mask]=background_freqs
    
    pdb.set_trace()
    
