from sys import argv
from collections import OrderedDict
import os
import time
from kerasAC.train import * 
from utils import *

mm10_chroms = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chrX','chrY']
mm10_splits = dict()
mm10_splits[0] = {'test':['chr1'],
                             'valid':['chr10','chr8']}
mm10_splits[1] = {'test':['chr19','chr2'],
                             'valid':['chr1']}
mm10_splits[2] = {'test':['chr3'],
                             'valid':['chr19','chr2']}
mm10_splits[3] = {'test':['chr13','chr6'],
                             'valid':['chr3']}
mm10_splits[4] = {'test':['chr5','chr16','chrY'],
                             'valid':['chr13','chr6']}
mm10_splits[5] = {'test':['chr4','chr15'],
                             'valid':['chr5','chr16','chrY']}
mm10_splits[6] = {'test':['chr7','chr18','chr14'],
                             'valid':['chr4','chr15']}
mm10_splits[7] = {'test':['chr11','chr17','chrX'],
                             'valid':['chr7','chr18','chr14']}
mm10_splits[8] = {'test':['chr12','chr9'],
                             'valid':['chr11','chr17','chrX']}
mm10_splits[9] = {'test':['chr10','chr8'],
                             'valid':['chr12','chr9']}




params = argv[1:]
gpuno = int(params[0])
splitno = int(params[1])

root = "/mnt/lab_data2/msharmin/oc-atlas"
logger = get_logger('{}/logs/train_multi_binary_gc{}.log'.format(root, splitno))

def run_tiledb_multi_classification_model():
    split = splitno
    params = {}
    params["ref_fasta"] = "/mnt/data/pipeline_genome_data/mm10/mm10_no_alt_analysis_set_ENCODE.fasta"
    params["chrom_sizes"] = "/mnt/data/pipeline_genome_data/mm10/mm10.chrom.sizes"    
    params["tdb_indexer"] = "task.tiledb.kerasAC.tsv"
    params["tdb_partition_attribute_for_upsample"] = "idr_peak"
    params["tdb_partition_thresh_for_upsample"] = 1    
    params["tdb_inputs"] = ["seq", "gc_inputs.tsv"]
    params["tdb_outputs"] = ["task.tiledb.kerasAC.tsv"]
    params["tdb_input_source_attribute"] = ["seq", "bigwig_track"]
    params["tdb_output_source_attribute"] = ["idr_peak"]
    params["tdb_input_flank"] = [500, 500]
    params["tdb_output_flank"] = [100]
    params["tdb_input_aggregation"] = [None,"mean"]
    params["tdb_input_transformation"] = [None, None]
    params["tdb_output_aggregation"] = ["max"]
    params["tdb_output_transformation"] = [None]
    params["num_inputs"] = 2
    params["num_outputs"] =1
    
    params["upsample_thresh_list_train"] = [0,0.1]
    params["upsample_thresh_list_eval"] = [0,0.1]
    params["upsample_ratio_list_train"] = [0.7]
    params["upsample_ratio_list_eval"] = [0.98]
    
    params["num_train"] = 100000
    params["num_valid"] = 100000
    params["num_tasks"] = 277
    params["threads"] = 20
    params["max_queue_size"] = 50
    params["patience"] = 3
    params["patience_lr"] = 2
    params["batch_size"] = 256
    params["train_chroms"] = list(set(mm10_chroms).difference(set(mm10_splits[split]["test"])).difference(set(mm10_splits[split]["valid"])))
    params["validation_chroms"] = mm10_splits[split]["valid"]
    params["architecture_from_file"] = "functional_basset_classification_gc_corrected.py"
    params["model_hdf5"] = "{}/models/matlas.functional_basset_binary_gc_corrected.{}".format(root, split)
    
    encode_model_root = "/mnt/lab_data/kundaje/users/dskim89/ggr/nn/models.npy_export.2019-04-15"
    params["init_weights"] = "{0}/encode-roadmap.basset.clf.testfold-{1}/encode-roadmap.model_variables.npz".format(encode_model_root, split)
    
    train(params) 
    return None


run_tiledb_multi_classification_model()
