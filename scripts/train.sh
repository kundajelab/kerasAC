#extra tasks for tf's that are enriched in false positive set (KLF4, OBOX5, PAX8, SPZ1, TCF3), initialize w/ ENCODE weights
CUDA_VISIBLE_DEVICES=7 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.train.bed \
		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.validate.bed \
		    --model_output_file easy.gc.1neg.1pos.tf_tasks \
		    --batch_size 1000 \
		    --num_tasks 6 \
		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
		    --architecture_spec basset_architecture_multitask_tfs.py

#extra tasks for tf's that are enriched in false positive set (KLF4, OBOX5, PAX8, SPZ1, TCF3), initialize w/ random weights
CUDA_VISIBLE_DEVICES=3 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.train.bed \
		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.validate.bed \
		    --model_output_file easy.gc.1neg.1pos.tf_tasks.randinit \
		    --batch_size 1000 \
		    --num_tasks 6 \
		    --architecture_spec basset_architecture_multitask_tfs.py


#CUDA_VISIBLE_DEVICES=7 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/gc.1.fp_control.shuffled.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.1.neg.1pos.validate.bed \
#		    --model_output_file easy.gc.1neg.1pos.fp_control \
#		    --batch_size 1000 \
#		    --num_tasks 1 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz
#
#
#CUDA_VISIBLE_DEVICES=3 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/gc.1.fp_control.shuffled.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.1.neg.1pos.validate.bed \
#		    --model_output_file easy.gc.1neg.1pos.fp_control.randinit \
#		    --batch_size 1000 \
#		    --num_tasks 1
#
#



#
#
#CUDA_VISIBLE_DEVICES=3 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.train.28.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.validate.28.bed \
#		    --test_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.test.28.bed \
#		    --model_output_file single.28.smallepoch \
#		    --batch_size 1000 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
#		    --num_tasks 1 \
#		    --num_train 5000 \
#		    --num_valid 10000 \
#		    --epochs 150 \
#		    --patience 150 \
#		    --architecture_spec basset_architecture_single_task.py
#
#
#CUDA_VISIBLE_DEVICES=1 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_negative_set/V576_DNase.train.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_negative_set/V576_DNase.validate.bed \
#		    --test_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/V576_DNase.test.bed \
#		    --model_output_file V576_gecco.smallepoch \
#		    --batch_size 1000 \
#		    --num_tasks 1 \
#		    --num_train 5000 \
#		    --num_valid 10000 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
#		    --epochs 150 \
#		    --patience 150
#
#
#CUDA_VISIBLE_DEVICES=7 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.1.neg.1pos.train.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.1.neg.1pos.validate.bed \
#		    --model_output_file easy.gc.1neg.1pos.smallepoch \
#		    --batch_size 1000 \
#		    --num_tasks 1 \
#		    --num_train 5000 \
#		    --num_valid 10000 \
#		    --epochs 200 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz
#
#
#
#
#CUDA_VISIBLE_DEVICES=7 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.5.neg.1pos.train.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.5.neg.1pos.validate.bed \
#		    --test_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.5.neg.1pos.test.bed \
#		    --model_output_file easy.gc.5neg.1pos.smallepoch \
#		    --batch_size 1000 \
#		    --num_tasks 1 \
#		    --num_train 5000 \
#		    --num_valid 10000 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
#		    --epochs 150 \
#		    --patience 150
#
#
#
#CUDA_VISIBLE_DEVICES=7 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.1neg.1pos.train.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.1neg.1pos.validate.bed \
#		    --test_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.1neg.1pos.test.bed \
#		    --model_output_file easy.dinuc.1neg.1pos.smallepoch \
#		    --batch_size 1000 \
#		    --num_tasks 1 \
#		    --num_train 5000 \
#		    --num_valid 10000 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
#		    --epochs 200 
#
#
#
#CUDA_VISIBLE_DEVICES=0 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.5.neg.1pos.train.bed \
#		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.5.neg.1pos.validate.bed \
#		    --test_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.5.neg.1pos.test.bed \
#		    --model_output_file easy.dinuc.5neg.1pos.smallepoch \
#		    --batch_size 1000 \
#		    --num_tasks 1 \
#		    --num_train 5000 \
#		    --num_valid 10000 \
#		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
#		    --epochs 150 \
#		    --patience 150
#
#
#
#
