import pdb 
from kerasAC.generators.basic_generator import *

index_path='DNASE.K562.regressionlabels.allbins.hdf5'
input_path=['seq','/srv/scratch/annashch/deeplearning/gc_experiments/k562/gc_hg38_110bp.hdf5']
output_path=['DNASE.K562.regressionlabels.allbins.hdf5']
num_inputs=2
num_outputs=1
ref_fasta="/mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
chroms_to_use=['chr1',
               'chr2',
               'chr3',
               'chr4',
               'chr5',
               'chr6',
               'chr8',
               'chr9',
               'chr10',
               'chr12',
               'chr13',
               'chr15',
               'chr16',
               'chr19',
               'chr20',
               'chr21',
               'chr22',
               'chrY']
upsample_thresh_list=[0,0.1]
upsample_ratio_list=[.7]
gen=DataGenerator(index_path=index_path,
                  input_path=input_path,
                  output_path=output_path,
                  num_inputs=num_inputs,
                  num_outputs=num_outputs,
                  ref_fasta=ref_fasta,
                  add_revcomp=False,
                  expand_dims=True,
                  shuffle=False,
                  upsample_thresh_list=upsample_thresh_list,
                  upsample_ratio_list=upsample_ratio_list,
                  chroms_to_use=chroms_to_use,
                  batch_size=1000,
                  return_coords=True)
#X,y,coords=gen[10]
#print(len(X))
#print(len(y))
#print(len(coords))
#print(X[0].shape)
#print(y[0].shape)

pdb.set_trace()

