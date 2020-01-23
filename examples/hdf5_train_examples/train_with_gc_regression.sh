#caprin is chrom 11,
#we  use split #7
CUDA_VISIBLE_DEVICES=3 kerasAC_train --index_data_path DNASE.K562.regressionlabels.allbins.hdf5 \
		    --input_data_path seq /srv/scratch/annashch/deeplearning/gc_experiments/k562/gc_hg38_110bp.hdf5 \
		    --output_data_path DNASE.K562.regressionlabels.allbins.hdf5 \
		    --upsample_thresh_list_train 0 0.1 \
		    --upsample_ratio_list_train 0.7 \
		    --upsample_thresh_list_eval 0 0.1 \
		    --upsample_ratio_list_eval 0.98 \
		    --num_inputs 2 \
		    --num_outputs 1 \
		    --model_hdf5 DNASE.K562.regressionlabels.7.withgc \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --batch_size 256 \
		    --train_chroms chr1 chr2 chr3 chr4 chr5 chr6 chr8 chr9 chr10 chr12 chr13 chr15 chr16 chr19 chr20 chr21 chr22  chrY \
		    --validation_chroms chr7 chr14 chr18 \
		    --architecture_from_file functional_basset_regression_gc_corrected.py \
		    --num_train 10000000 \
		    --num_valid 1000000 \
		    --num_tasks 1 \
		    --threads 10 \
		    --max_queue_size 500 \
		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
		    --patience 3 \
		    --patience_lr 2 \
		    --expand_dims
