import numpy as np
import pysam
import pandas as pd
import pdb
import tabix

def revcomp(seq):
    seq=seq[::-1].upper()
    comp_dict=dict()
    comp_dict['A']='T'
    comp_dict['T']='A'
    comp_dict['C']='G'
    comp_dict['G']='C'
    rc=[]
    for base in seq:
        if base in comp_dict:
            rc.append(comp_dict[base])
        else:
            rc.append(base)
    return ''.join(rc)

def add_variants(bed_entries,vcf,args,ltrdict):
    for seq_index in range(len(bed_entries)):
        bed_entry=bed_entries[seq_index]
        cur_start=bed_entry[1]
        cur_end=bed_entry[2]
        max_offset=cur_end-cur_start
        try:
            variants=[i for i in vcf.query(bed_entry[0],bed_entry[1],bed_entry[2])]
        except:
            continue
        if variants!=[]:
            for variant in variants:
                if args.global_vcf == False:
                    #the vcf contains personal variants for an individual
                    pos=float(variant[1])
                    offset=int(pos-cur_start)
                    if offset >= max_offset:
                        continue
                    alleles=variant[3].split(',')+variant[4].split(',')
                    geno=[int(i) for i in variant[9].split(':')[0].split('/')]
                    try:
                        freqs1=np.asarray(ltrdict[alleles[geno[0]]])
                        freqs2=np.asarray(ltrdict[alleles[geno[1]]])
                    except:
                        continue
                    freqs=np.mean((freqs1,freqs2),axis=0)

                else:
                    #common allele freqs for the reference genome
                    pos=float(variant[1])
                    offset=int(pos-cur_start)
                    if offset >= max_offset:
                        continue
                    alleles=[variant[3],variant[4][0]]
                    try:
                        all_freqs=[float(variant[5]),float(variant[6])]
                    except:
                        continue
                    freqs=np.asarray(ltrdict[alleles[0]])*all_freqs[0]+np.asarray(ltrdict[alleles[1]])*all_freqs[1]
                seqs[seq_index][offset]=freqs
    return seqs

#currently we have on-the-fly batch generation from hdf5 & bed files.
def data_generator(data_path,args,upsample_ratio):
    if data_path.endswith('.hdf5'):
        return data_generator_hdf5(data_path,args)
    elif (data_path.endswith('.bed') or data_path.endswith('.bed.gz') or data_path.endswith('.tsv') or data_path.endswith('.tsv.gz')):
      if upsample_ratio <= 0.0:
        return data_generator_bed_original(data_path, args)
      else:
        return data_generator_bed_upsample(data_path, args, upsample_ratio)
    else:
        raise Exception("data for generator must be in hdf5 format (.hdf5 0ending) or bed format (.bed ending). Neither is true. Exiting")

def data_generator_bed_upsample(bed_source, args, upsample_ratio):
    #open the reference file
    ref=pysam.FastaFile(args.ref_fasta)
    #load the train data as a pandas dataframe, skip the header
    data=pd.read_csv(bed_source,header=0,sep='\t',index_col=[0,1,2])
    ones = data.loc[(data > 0).any(axis=1)]
    zeros = data.loc[(data < 1).all(axis=1)]
    #decide if reverse complement should be used
    if args.revcomp==True:
        batch_size=args.batch_size/2
    else:
        batch_size=args.batch_size
    pos_batch_size = int(args.batch_size * upsample_ratio)
    neg_batch_size = args.batch_size - pos_batch_size
    ltrdict = {'a':[1,0,0,0],
               'c':[0,1,0,0],
               'g':[0,0,1,0],
               't':[0,0,0,1],
               'n':[0,0,0,0],
               'A':[1,0,0,0],
               'C':[0,1,0,0],
               'G':[0,0,1,0],
               'T':[0,0,0,1],
               'N':[0,0,0,0]}
    #if vcf file is provided, load the subject's variants
    vcf=None
    if (args.vcf_file!=None):
        vcf=tabix.open(args.vcf_file)
    #iterate through batches and one-hot-encode on the fly
    pos_start_index = 0
    pos_num_generated = 0
    pos_total_entries = ones.shape[0] - pos_batch_size
    neg_start_index = 0
    neg_num_generated = 0
    neg_total_entries = zeros.shape[0] - neg_batch_size
    while True:
        if (pos_num_generated >= pos_total_entries):
            pos_start_index=0
            ones = pd.concat([ones[pos_num_generated:], ones[:pos_num_generated]])
            pos_num_generated = 0
        if (neg_num_generated >= neg_total_entries):
            neg_start_index = 0
            zeros = pd.concat([zeros[neg_num_generated:], zeros[:neg_num_generated]])
            neg_num_generated = 0
        pos_end_index = pos_start_index + int(pos_batch_size)
        neg_end_index = neg_start_index + int(neg_batch_size)
        #get seq positions
        pos_bed_entries=[(ones.index[i]) for i in range(pos_start_index,pos_end_index)]
        neg_bed_entries=[(zeros.index[i]) for i in range(neg_start_index, neg_end_index)]
        bed_entries = pos_bed_entries + neg_bed_entries
        #get sequences
        seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if args.revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        #add in subject-specific allele frequencies, if provided
        if vcf!=None:
            seqs=add_variants(bed_entries,vcf,args,ltrdict)
        #expand dimension of 1,  unless we're dealing with a GRU in recurrent network
        if(args.squeeze_input_for_gru==False):
            x_batch=np.expand_dims(seqs,1)
        else:
            x_batch=seqs
        y_labels_ones = ones[pos_start_index:pos_end_index]
        y_labels_zeros = zeros[neg_start_index:neg_end_index]
        y_batch_ones = np.asarray(y_labels_ones)
        y_batch_zeros = np.asarray(y_labels_zeros)
        y_batch = np.concatenate([y_batch_ones, y_batch_zeros])
        if args.revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        pos_num_generated += pos_batch_size
        neg_num_generated += neg_batch_size
        pos_start_index = pos_end_index
        neg_start_index = neg_end_index
        if (args.squeeze_input_for_gru==False):
            if ((x_batch.ndim < 4) or (y_batch.ndim <2)):
                print("skipping!: hint-- is your reference (i.e. hg19) correct?")
                continue
            else:
                yield tuple([x_batch,y_batch])
        else:
            if ((x_batch.ndim <3) or (y_batch.ndim <2)):
                print("skipping!: hint -- is your reference (i.e. hg19) correct?")
                continue
            else:
                yield tuple([x_batch,y_batch])

def data_generator_bed_original(bed_source,args):
    #open the reference file
    ref=pysam.FastaFile(args.ref_fasta)
    #load the train data as a pandas dataframe, skip the header
    data=pd.read_csv(bed_source,header=0,sep='\t',index_col=[0,1,2])
    ltrdict = {'a':[1,0,0,0],
               'c':[0,1,0,0],
               'g':[0,0,1,0],
               't':[0,0,0,1],
               'n':[0,0,0,0],
               'A':[1,0,0,0],
               'C':[0,1,0,0],
               'G':[0,0,1,0],
               'T':[0,0,0,1],
               'N':[0,0,0,0]}
    #if vcf file is provided, load the subject's variants
    vcf=None
    if (args.vcf_file!=None):
        vcf=tabix.open(args.vcf_file)
    #iterate through batches and one-hot-encode on the fly
    start_index=0
    num_generated=0
    total_entries=data.shape[0]-args.batch_size
    #decide if reverse complement should be used
    if args.revcomp==True:
        batch_size=args.batch_size/2
    else:
        batch_size=args.batch_size
    while True:
        if (num_generated >=total_entries):
            start_index=0
        end_index=start_index+int(batch_size)
        #get seq positions
        bed_entries=[(data.index[i]) for i in range(start_index,end_index)]
        #get sequences
        seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if args.revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        #add in subject-specific allele frequencies, if provided
        if vcf!=None:
            seqs=add_variants(bed_entries,vcf,args,ltrdict)
        #expand dimension of 1,  unless we're dealing with a GRU in recurrent network
        if(args.squeeze_input_for_gru==False):
            x_batch=np.expand_dims(seqs,1)
        else:
            x_batch=seqs
        y_batch=np.asarray(data[start_index:end_index])
        if args.revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        num_generated+=batch_size
        start_index=end_index
        if (args.squeeze_input_for_gru==False):
            if ((x_batch.ndim < 4) or (y_batch.ndim <2)):
                #pdb.set_trace() 
                print("skipping!: hint-- is your reference (i.e. hg19) correct?")
                continue
            else:
                yield tuple([x_batch,y_batch])
        else:
            if ((x_batch.ndim <3) or (y_batch.ndim <2)):
                print("skipping!: hint -- is your reference (i.e. hg19) correct?")
                continue
            else:
                yield tuple([x_batch,y_batch])

def data_generator_hdf5(data_path,args):
    hdf5_source=h5py.File(data_path,'r')
    num_generated=0
    total_entries=hdf5_source['X']['default_input_mode_name'].shape[0]
    start_index=0
    batch_size=args.batch_size
    while True:
        if(num_generated >=total_entries):
            start_index=0
        end_index=start_index+batch_size
        x_batch=hdf5_source['X']['default_input_mode_name'][start_index:end_index]
        y_batch=hdf5_source['Y']['default_output_mode_name'][start_index:end_index]
        num_generated+=batch_size
        start_index=end_index
        yield tuple([x_batch,y_batch])
