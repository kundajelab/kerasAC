#!/bin/bash

#fold to use for training 
fold=$1

#gpu to use for training 
gpu=$2

#create a title for the model
model_name=$3

#set seed for training
if [ -z "$4" ]
then
    seed=1234
else
    seed=$4
fi
echo "seed:$seed"

#output directory 
if [ -z "$5" ]
then
    outdir='.'
else
    outdir=$5
fi
echo "outdir:$outdir"
CUDA_VISIBLE_DEVICES=$gpu python train.py \
		    --seed $seed \
		    --revcomp \
		    --batch_size 5 \
                    --datasets H3K4me3 \
		    --ref_fasta /mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
                    --tdb_array /mnt/lab_data2/kmualim/ChromAtlas_BPNet/chip-seq-doc/db_H3K4me3/ \
		    --tdb_partition_attribute_for_upsample overlap_peak \
		    --tdb_partition_thresh_for_upsample 1 \
		    --tdb_input_source_attribute seq control_count_bigwig_plus_5p,control_count_bigwig_minus_5p control_count_bigwig_plus_5p,control_count_bigwig_minus_5p \
		    --tdb_input_aggregation None None,None sum,sum \
		    --tdb_input_transformation None None,None log,log \
		    --tdb_input_flank 35500 2500,2500 2500,2500 \
		    --tdb_output_source_attribute count_bigwig_plus_5p,count_bigwig_minus_5p count_bigwig_plus_5p,count_bigwig_minus_5p \
		    --tdb_output_flank 2500,2500 2500,2500 \
		    --tdb_output_aggregation None,None sum,sum \
		    --tdb_output_transformation None,None log,log \
		    --tdb_input_min None None,None None,None \
		    --tdb_input_max None None,None None,None \
		    --tdb_output_min None,None 4.6,4.6 \
		    --tdb_output_max None,None 11.5,11.5 \
		    --fold $fold \
		    --genome hg38 \
		    --num_train 10000 \
		    --num_valid 10000 \
		    --upsample_threads 24 \
		    --threads 0 \
		    --max_queue_size 20 \
		    --patience 6 \
		    --patience_lr 3 \
		    --model_prefix $outdir/$model_name.$fold \
		    --model_params params.txt \
                    --num_tasks 2 \
                    --num_inputs 3 \
                    --num_outputs 2 \
                    --architecture_spec profile_bpnet_chipseq \
                    --tdb_input_datasets seq H3K4me3,H3K4me3 H3K4me3,H3K4me3 \
                    --tdb_output_datasets H3K4me3,H3K4me3 H3K4me3,H3K4me3 \
		    --upsample_ratio_list_train 1.0 \
		    --upsample_ratio_list_eval 1.0 \
                    --tdb_partition_attribute_for_upsample overlap_peak \
                    --tdb_partition_thresh_for_upsample 1 \
                    --tdb_partition_datasets_for_upsample H3K4me3 \
		    --trackables logcount_predictions_loss loss profile_predictions_loss val_logcount_predictions_loss val_loss val_profile_predictions_loss \
                    --bed_regions /mnt/lab_data2/kmualim/ChromAtlas_BPNet/ENCSR087PFU/data/peaks.bed.gz \
                    --bed_regions_center random \
                    --bed_regions_jitter 100 \
#                    --conv1_kernel_size 25.0
#		    --tdb_ambig_attribute ambig_peak \
