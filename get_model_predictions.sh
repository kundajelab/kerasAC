#STARTING PREDICTIONS
for cur_model in easy.gc.1neg.1pos.tf_tasks easy.gc.1neg.1pos.tf_tasks.randinit 
do
CUDA_VISIBLE_DEVICES=3 python get_model_predictions.py --model_hdf5 $cur_model --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/tasks_for_difficult_tfs/multi_task.test.bed  --predictions_pickle $cur_model.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file $cur_model.acc
done


#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 easy.gc.1neg.1pos.starting  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.1.neg.1pos.test.bed --predictions_pickle easy.gc.1neg.1pos.starting.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.gc.1neg.1pos.starting.acc
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 easy.gc.5neg.1pos.starting  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.5.neg.1pos.test.bed --predictions_pickle easy.gc.5neg.1pos.starting.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.gc.5neg.1pos.starting.acc
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 easy.dinuc.1neg.1pos.starting  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.1neg.1pos.test.bed --predictions_pickle easy.dinuc.1neg.1pos.starting.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.dinuc.1neg.1pos.starting.acc &
#
#CUDA_VISIBLE_DEVICES=3 python get_model_predictions.py --model_hdf5 easy.dinuc.5neg.1pos.starting  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.5.neg.1pos.test.bed --predictions_pickle easy.dinuc.5.neg.1pos.starting.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.dinuc.5neg.1pos.starting.acc &
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 single.28.starting  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.test.28.bed --predictions_pickle single.28.starting.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  single.28.starting.acc
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 V576_gecco.starting  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_negative_set/V576_DNase.test.bed --predictions_pickle V576_gecco.starting.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  V576_gecco.starting.acc
#
#
##SMALL EPOCH PREDICTIONS
#
#CUDA_VISIBLE_DEVICES=1 python get_model_predictions.py --model_hdf5 easy.gc.1neg.1pos.randinit.smallepoch  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.1.neg.1pos.test.bed --predictions_pickle easy.gc.1neg.1pos.randinit.smallepoch.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.gc.1neg.1pos.randinit.smallepoch.acc
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 easy.gc.5neg.1pos.smallepoch   --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.gc.5.neg.1pos.test.bed --predictions_pickle easy.gc.5neg.1pos.smallepoch.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.gc.5neg.1pos.smallepoch.acc
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 easy.dinuc.1neg.1pos.smallepoch  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.1neg.1pos.test.bed --predictions_pickle easy.dinuc.1neg.1pos.smallepoch.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.dinuc.1neg.1pos.smallepoch.acc &
#
#CUDA_VISIBLE_DEVICES=7 python get_model_predictions.py --model_hdf5 easy.dinuc.5neg.1pos.smallepoch  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_easy_set/easy.V576.dinuc.5.neg.1pos.test.bed --predictions_pickle easy.dinuc.5.neg.1pos.smallepoch.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  easy.dinuc.5neg.1pos.smallepoch.acc
#
#
# 
#CUDA_VISIBLE_DEVICES=3 python get_model_predictions.py --model_hdf5 single.28.smallepoch  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/archive/single.test.28.bed --predictions_pickle single.28.smallepoch.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  single.28.smallepoch.acc
#
#
#CUDA_VISIBLE_DEVICES=0 python get_model_predictions.py --model_hdf5 V576_gecco.smallepoch  --data_bed /srv/scratch/annashch/deeplearning/form_inputs/gecco_inputs_v2/experiments_negative_set/V576_DNase.test.bed --predictions_pickle V576_gecco.smallepoch.vars --batch_size 1000 --sequential --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa --accuracy_metrics_file  V576_gecco.smallepoch.acc
#
