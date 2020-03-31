#referenced files are in /users/annashch/kerasAC/examples/hdf5_predict_examples
CUDA_VISIBLE_DEVICES=2 kerasAC_predict_hdf5 --index_data_path DNASE.K562.regressionlabels.allbins.hdf5 \
		    --input_data_path seq /srv/scratch/annashch/deeplearning/gc_experiments/k562/gc_hg38_110bp.hdf5 \
		    --output_data_path  DNASE.K562.regressionlabels.allbins.hdf5 \
		    --num_inputs 2 \
		    --num_outputs 1 \
		    --ref_fasta /mnt/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --load_model_hdf5 DNASE.K562.regressionlabels.withgc.0.hdf5 \
		    --batch_size 1000 \
		    --predict_chroms chrY \
		    --threads 10 \
		    --max_queue_size 100 \
		    --predictions_and_labels_hdf5 gc.dnase.k562 \
		    --calibrate_regression \
		    --expand_dims

