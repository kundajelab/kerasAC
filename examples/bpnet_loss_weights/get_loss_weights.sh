kerasAC_loss_weights_bpnet  --tdb_array /oak/stanford/groupa/akundaje/projects/atlas/tiledb/tier1/dnase \
			    --chroms chr1 \
			    --upsample_attribute overlap_peak \
			    --label_attribute count_bigwig_unstranded_5p \
			    --num_threads 1 \
			    --task ENCSR000EOT \
			    --upsample_thresh 1 \
			    --flank 500
