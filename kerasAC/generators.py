from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
import math 
import pysam
from .util import ltrdict
import threading
import pickle
import pdb

def get_weights(data):
    w1=[float(data.shape[0])/sum(data.iloc[:,i]==1) for i in range(data.shape[1])]
    w0=[float(data.shape[0])/sum(data.iloc[:,i]==0) for i in range(data.shape[1])]
    return w1,w0
    

def dinuc_shuffle(seq):
    #get list of dinucleotides
    nucs=[]
    for i in range(0,len(seq),2):
        nucs.append(seq[i:i+2])
    #generate a random permutation
    random.shuffle(nucs)
    return ''.join(nucs) 


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

def get_probability_thresh_for_precision(truth,predictions,precision_thresh):
    from sklearn.metrics import precision_recall_curve
    num_tasks=truth.shape[1]
    precision_thresholds=[] 
    for task_index in range(num_tasks):
        truth_task=truth.iloc[:,task_index]
        pred_task=predictions[:,task_index]
        non_ambig=truth_task!=-1        
        precision,recall,threshold=precision_recall_curve(truth_task[non_ambig],pred_task[non_ambig])
        threshold=np.insert(threshold,threshold.shape[0],1)
        merged_prc=pd.DataFrame({'precision':precision,
                                 'recall':recall,
                                 'threshold':threshold})
        precision_thresholds.append(np.min(merged_prc[merged_prc['precision']>=precision_thresh]['threshold']))
    print(precision_thresholds) 
    return precision_thresholds
    
def open_data_file(data_path,tasks,chroms_to_use):
    if data_path.endswith('.hdf5'):
        if tasks==None:
            data=pd.read_hdf(data_path)
        else:
            data=pd.read_hdf(data_path,columns=tasks)
    else:
        #treat as bed file 
        if tasks==None:
            data=pd.read_csv(data_path,header=0,sep='\t',index_col=[0,1,2])
        else:
            data=pd.read_csv(data_path,header=0,sep='\t',nrows=1)
            chrom_col=data.columns[0]
            start_col=data.columns[1]
            end_col=data.columns[2]
            data=pd.read_csv(data_path,header=0,sep='\t',usecols=[chrom_col,start_col,end_col]+tasks,index_col=[0,1,2])
    print("loaded labels") 
    if chroms_to_use!=None:
        data=data[np.in1d(data.index.get_level_values(0), chroms_to_use)]
    print("filtered on chroms_to_use")
    return data 

class TruePosGenerator(Sequence):
    def __init__(self,data_pickle,ref_fasta,batch_size=128,precision_thresh=0.9):
        f=open(data_pickle,'rb')
        data=pickle.load(f)
        self.predictions=data[0]
        self.labels=data[1]
        self.columns=self.labels.columns
        #calculate prediction probability cutoff to achieve the specified precision threshold
        self.prob_thresholds=get_probability_thresh_for_precision(self.labels,self.predictions,precision_thresh)
        truth_pred_product=self.labels*(self.predictions>=self.prob_thresholds)
        true_pos_rows=truth_pred_product[truth_pred_product.max(axis=1)>0]
        self.data=true_pos_rows
        self.indices=np.arange(self.data.shape[0])
        self.add_revcomp=False
        self.ref_fasta=ref_fasta
        self.lock=threading.Lock()
        self.batch_size=batch_size

        
    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            self.ref=pysam.FastaFile(self.ref_fasta)
            return self.get_basic_batch(idx)
        
    def get_basic_batch(self,idx):
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.data.index[inds]
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        #extract the labels at the current batch of indices 
        y_batch=np.asarray(self.data.iloc[inds])
        return (bed_entries,x_batch,y_batch)    


#use wrappers for keras Sequence generator class to allow batch shuffling upon epoch end
class DataGenerator(Sequence):
    def __init__(self,data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None,shuffled_ref_negatives=False,upsample=True,upsample_ratio=0.1,chroms_to_use=None,get_w1_w0=False):
        self.lock = threading.Lock()        
        self.batch_size=batch_size
        #decide if reverse complement should be used
        self.add_revcomp=add_revcomp
        if add_revcomp==True:
            self.batch_size=int(batch_size/2)

        #determine whether negative set should consist of the shuffled refs.
        # If so, split batch size in 2, as each batch will be augmented with shuffled ref negatives
        # in ratio equal to positives 
        self.shuffled_ref_negatives=shuffled_ref_negatives
        if self.shuffled_ref_negatives==True:
            self.batch_size=int(self.batch_size/2)
    
        #open the reference file
        self.ref_fasta=ref_fasta

        self.data=open_data_file(data_path,tasks,chroms_to_use)
        if get_w1_w0==True:
            w1,w0=get_weights(self.data)
            self.w1=w1
            self.w0=w0
            print(self.w1)
        self.indices=np.arange(self.data.shape[0])
        num_indices=self.indices.shape[0]
        self.add_revcomp=add_revcomp
        
        #set variables needed for upsampling the positives
        self.upsample=upsample
        if self.upsample==True:
            self.upsample_ratio=upsample_ratio
            self.ones = self.data.loc[(self.data > 0).any(axis=1)]
            self.zeros = self.data.loc[(self.data < 1).all(axis=1)]
            self.pos_batch_size = int(self.batch_size * self.upsample_ratio)
            self.neg_batch_size = self.batch_size - self.pos_batch_size
            self.pos_indices=np.arange(self.ones.shape[0])
            self.neg_indices=np.arange(self.zeros.shape[0])
            
            #wrap the positive and negative indices to reach size of self.indices
            num_pos_wraps=math.ceil(num_indices/self.pos_indices.shape[0])
            num_neg_wraps=math.ceil(num_indices/self.neg_indices.shape[0])
            self.pos_indices=np.repeat(self.pos_indices,num_pos_wraps)[0:num_indices]
            np.random.shuffle(self.pos_indices)
            self.neg_indices=np.repeat(self.neg_indices,num_neg_wraps)[0:num_indices]
            np.random.shuffle(self.neg_indices)
            
    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            ref=pysam.FastaFile(self.ref_fasta)
            self.ref=ref
            if self.shuffled_ref_negatives==True:
                return self.get_shuffled_ref_negatives_batch(idx)
            elif self.upsample==True:
                return self.get_upsampled_positives_batch(idx)
            else:
                return self.get_basic_batch(idx) 
        
    def get_shuffled_ref_negatives_batch(self,idx): 
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.data.index[inds]
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if self.add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
            
        #generate the corresponding negative set by dinucleotide-shuffling the sequences
        seqs_shuffled=[dinuc_shuffle(s) for s in seqs]
        seqs=seqs+seqs_shuffled
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        y_batch=np.asarray(self.data.iloc[inds])
        if self.add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        y_shape=y_batch.shape 
        y_batch=np.concatenate((y_batch,np.zeros(y_shape)))
        return (x_batch,y_batch)

    def get_upsampled_positives_batch(self,idx):
        #get seq positions
        pos_inds=self.pos_indices[idx*self.pos_batch_size:(idx+1)*self.pos_batch_size]
        pos_bed_entries=self.ones.index[pos_inds]
        neg_inds=self.neg_indices[idx*self.neg_batch_size:(idx+1)*self.neg_batch_size]
        neg_bed_entries=self.zeros.index[neg_inds]
    
        #print(neg_inds[0:10])
        #bed_entries=pos_bed_entries+neg_bed_entries

        #get sequences
        pos_seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in pos_bed_entries]
        neg_seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in neg_bed_entries]
        seqs=pos_seqs+neg_seqs 
        if self.add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
            
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        
        #extract the positive and negative labels at the current batch of indices
        y_batch_pos=self.ones.iloc[pos_inds]
        y_batch_neg=self.zeros.iloc[neg_inds]
        y_batch=np.concatenate((y_batch_pos,y_batch_neg),axis=0)
        #add in the labels for the reverse complement sequences, if used 
        if self.add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        return (x_batch,y_batch)            
    
    def get_basic_batch(self,idx):
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.data.index[inds]
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if self.add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        #extract the labels at the current batch of indices 
        y_batch=np.asarray(self.data.iloc[inds])
        #add in the labels for the reverse complement sequences, if used 
        if self.add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        return (x_batch,y_batch)
    
    def on_epoch_end(self):
        #if upsampling is being used, shuffle the positive and negative indices 
        if self.upsample==True:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)
        else:
            np.random.shuffle(self.indices)
