kerasAC_loss_weights_bpnet --tdb_array /mnt/lab_data2/kmualim/ChromAtlas_BPNet/chip-seq-doc/db_H3K4me3/ \
			    --chroms chr1 \
			    --upsample_attribute overlap_peak \
			    --label_attribute count_bigwig_plus_5p \
			    --num_threads 1 \
			    --task  H3K4me3 \
			    --upsample_thresh 1 \
			    --flank 2500
