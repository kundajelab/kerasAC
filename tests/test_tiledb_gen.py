from kerasAC.tiledb_generators import *
import pdb 
tdbgen=TiledbGenerator(shuffle=True,
                       batch_size=100,
                       chrom_sizes="/mnt/data/tiledb/hg38.chrom.sizes",
                       task_file="tasks.tsv",
                       label_source='fc_bigwig',
                       label_flank=3000,
                       label_aggregation=None,
                       sequence_flank=6500,
                       partition_attribute_for_upsample='idr_peak',
                       partition_thresh_for_upsample=1,
                       fraction_to_upsample=0.3)
print(len(tdbgen))
print(tdbgen.upsampled_indices_len)
print(tdbgen.non_upsampled_indices_len)
pdb.set_trace() 
