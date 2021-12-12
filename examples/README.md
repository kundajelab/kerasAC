The examples here utilize a tiledb dnase database of ENCODE tier 1 cell lines generated as follows: 

```
db_ingest --tiledb_metadata tier1.encode.dnase.tasks.tsv \
	  --array_name /oak/stanford/groups/akundaje/projects/atlas/tiledb/tier1/dnase \
	  --overwrite \
	  --chrom_sizes hg38.chrom.sizes \
	  --attribute_config encode_pipeline \
	  --coord_tile_size 10000 \
	  --task_tile_size 1 \
	  --write_chunk 30000000 \
	  --threads 20 \
	  --max_queue_size 50 \
	  --max_mem_g 200
```

Many of the examples use the K562 (ENCSR000EOT) dataset within this tiledb database.
