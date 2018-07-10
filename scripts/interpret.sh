CUDA_VISIBLE_DEVICES=7 kerasAC_interpret --model_hdf5 single.0 \
       --w0 1.10 \
       --w1 10.99 \
       --input_bed /srv/scratch/annashch/gecco/high.conf.variants.bed \
       --outf dl.scores \
       --ref /srv/scratch/annashch/deeplearning/form_inputs/code/hg19.genome.fa
