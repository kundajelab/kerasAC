#!/bin/bash
#--train_chroms chr2 chr3 chr4 chr5 chr6 chr7 chr9 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY \
#--validation_chroms chr8 chr10 \
#--train_chroms chr21 \
#--validation_chroms chr22 \

#export OMP_NUM_THREADS=1
#export USE_SIMPLE_THREADED_LEVEL3=1
CUDA_VISIBLE_DEVICES=0 kerasAC_train --tiledb_tasks_file tasks.tsv \
		    --chrom_sizes /mnt/data/annotations/by_release/hg38/hg38.chrom.sizes \
		    --train_chroms chr2 chr3 chr4 chr5 chr6 chr7 chr9 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY \
		    --validation_chroms chr8 chr10 \
		    --batch_size 20 \
                    --label_source fc_bigwig \
                    --label_flank 1500 \
                    --label_aggregation None \
                    --sequence_flank 6500 \
		    --partition_attribute_for_upsample idr_peak \
		    --partition_thresh_for_upsample 1 \
		    --train_upsample 0.3 \
		    --valid_upsample 0 \
		    --num_train 10000 \
		    --num_valid 10000 \
		    --num_tasks 1 \
		    --threads 40 \
		    --max_queue_size 10 \
		    --patience 3 \
		    --patience_lr 2 \
		    --model_hdf5 ATAC.pseudobulk.ADPD.Cluster24.profile.0 \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --architecture_spec profile_model
