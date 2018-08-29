#example 1 
CUDA_VISIBLE_DEVICES=7 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.train.bed \
		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.validate.bed \
		    --model_output_file easy.gc.1neg.1pos.tf_tasks \
		    --batch_size 1000 \
		    --num_tasks 6 \
		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
		    --architecture_spec basset_architecture_multitask_tfs
#example 2
CUDA_VISIBLE_DEVICES=3 kerasAC_train --train_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.train.28.bed \
		    --valid_path /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.validate.28.bed \
		    --model_output_file single.28.smallepoch \
		    --batch_size 1000 \
		    --init_weights /srv/scratch/annashch/deeplearning/encode-roadmap.dnase_tf-chip.batch_256.params.npz \
		    --num_tasks 1 \
		    --num_train 5000 \
		    --num_valid 10000 \
		    --epochs 150 \
		    --patience 150 \
		    --architecture_spec basset_architecture_single_task


