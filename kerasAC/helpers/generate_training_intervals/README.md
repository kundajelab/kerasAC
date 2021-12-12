This contains utility scripts to:
1) `bed_to_model_input_windows.py` : go from a bed file to a bed file containing peak intervals that span the full peak size  
2) `get_gc_content.py` : generate a pickled file of gc_content -> training region (or negative region)  
3) `get_gc_matched_ngatives.py` : select n negatives for each positive with matched gc content   

The universal negative gc pickles for ATAC & DNASE might be useful here:
* /oak/stanford/groups/akundaje/refs/backgrounds_accesibility/DNASE_background_hg38/universal_negatives_dnase.gc.candidates.pkl

* /oak/stanford/groups/akundaje/refs/backgrounds_accesibility/ATAC_background_hg38/universal_negatives_atac.gc.candidates.pkl 

see example for generating GM12878 ATAC & DNASE gc-matched negative set, here:
https://github.com/kundajelab/chrombpnet/tree/master/make_negatives

