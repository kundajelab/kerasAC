#generates hg19_splits for cross-validation
#hg19 (i.e. /mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes). Any chromosome from this chrom.sizes file that is not in the test/validation split is assumed to be in the training split (only considering chroms 1 - 22, X, Y
hg19_chroms=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX','chrY']
hg19_splits=dict()
hg19_splits[0]={'test':['chr1'],
           'valid':['chr10','chr8']}
hg19_splits[1]={'test':['chr19','chr2'],
           'valid':['chr1']}
hg19_splits[2]={'test':['chr3','chr20'],
           'valid':['chr19','chr2']}
hg19_splits[3]={'test':['chr13','chr6','chr22'],
           'valid':['chr3','chr20']}
hg19_splits[4]={'test':['chr5','chr16','chrY'],
           'valid':['chr13','chr6','chr22']}
hg19_splits[5]={'test':['chr4','chr15','chr21'],
           'valid':['chr5','chr16','chrY']}
hg19_splits[6]={'test':['chr7','chr18','chr14'],
           'valid':['chr4','chr15','chr21']}
hg19_splits[7]={'test':['chr11','chr17','chrX'],
           'valid':['chr7','chr18','chr14']}
hg19_splits[8]={'test':['chr12','chr9'],
           'valid':['chr11','chr17','chrX']}
hg19_splits[9]={'test':['chr10','chr8'],
           'valid':['chr12','chr9']}


#Note: the splits for hg19 and hg38 are the same, as are the chromosomes used for training models. 
splits=dict()
splits['hg19']=hg19_splits 
splits['hg38']=hg19_splits
chroms=dict()
chroms['hg19']=hg19_chroms
chroms['hg38']=hg19_chroms
