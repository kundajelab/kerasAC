#!/bin/bash
db_ingest --tiledb_metadata combine_inputs.tsv \
          --array_name chip_and_dnase_db_H3K27Ac \
          --chrom_sizes /users/annashch/hg38.chrom.sizes \
          --attribute_config_file combine_attribs.txt \
          --coord_tile_size 10000 \
          --task_tile_size 1 \
          --write_chunk 10000000 \
          --threads 40 \
          --max_queue_size 100 \
          --max_mem_g 500
