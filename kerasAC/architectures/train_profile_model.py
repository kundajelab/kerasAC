import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
import pyBigWig

from Bio.Seq import Seq
from pyfaidx import Fasta
import math

#not train on the peak summit as wavenet doesnt have any positional abilities. 

fasta_seq=Fasta('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/GRCh38.p3.genome.fa')
dnase_bigwig_hg38=pyBigWig.open('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/k562_hg38_DNASE_fc.signal.bigwig')
h3k27ac_bigwig_hg38=pyBigWig.open('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/k562_hg38_h3k27ac.bigWig')
dnase_overlap_reproducibility_peaks=pd.read_csv('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/Overlap_Reproducibility_peaks.narrowPeak',sep='\t',header=None)
dnase_overlap_reproducibility_peaks=dnase_overlap_reproducibility_peaks[[0,1,2]]
dnase_overlap_reproducibility_peaks=dnase_overlap_reproducibility_peaks.drop_duplicates()
dnase_overlap_reproducibility_peaks['peak_length']=dnase_overlap_reproducibility_peaks[2]-dnase_overlap_reproducibility_peaks[1]
dnase_overlap_reproducibility_peaks['absolute_diff']=3000-dnase_overlap_reproducibility_peaks['peak_length']
dnase_overlap_reproducibility_peaks['left_req_extension']=dnase_overlap_reproducibility_peaks['absolute_diff'].apply(lambda x : math.floor(float(x)/2))
dnase_overlap_reproducibility_peaks['right_req_extension']=dnase_overlap_reproducibility_peaks['absolute_diff'].apply(lambda x : math.ceil(float(x)/2))
dnase_overlap_reproducibility_peaks['L_left_pos']=dnase_overlap_reproducibility_peaks[1]-dnase_overlap_reproducibility_peaks['left_req_extension']
dnase_overlap_reproducibility_peaks['L_right_pos']=dnase_overlap_reproducibility_peaks[2]+dnase_overlap_reproducibility_peaks['right_req_extension']
dnase_overlap_reproducibility_peaks['new_diff']=dnase_overlap_reproducibility_peaks['L_right_pos']-dnase_overlap_reproducibility_peaks['L_left_pos']
dnase_overlap_reproducibility_peaks['L_left_pos']=dnase_overlap_reproducibility_peaks['L_left_pos'].astype(int)
dnase_overlap_reproducibility_peaks['L_right_pos']=dnase_overlap_reproducibility_peaks['L_right_pos'].astype(int)
#dnase_overlap_reproducibility_peaks[[0,'L_left_pos','L_right_pos']].to_csv('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/dnase_peak_list.bed',sep='\t',header=None,index=False)
#os.system('bedtools makewindows -g /oak/stanford/groups/akundaje/laks/proseq_chromputer/hg19.chrom.sizes -w 2000 -s 200 > /oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/genome.10kb.bed')
#os.system('bedtools subtract -a /oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/genome.10kb.bed -b /oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/dnase_peak_list.bed > /oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/genome.10kb_h3k27ac_k562_hg38_negatives.bed')

#preparing the positive peak file
dnase_positives=dnase_overlap_reproducibility_peaks[[0,'L_left_pos','L_right_pos']]
dnase_positives.columns=['chr','start','end']
dnase_positives_validation=dnase_positives[dnase_positives['chr']=='chr22']
dnase_positives_test=dnase_positives[dnase_positives['chr']=='chr19']
dnase_positives=dnase_positives[dnase_positives['chr'].isin(['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr20','chr21','chrX'])]
dnase_positives['id']=dnase_positives['chr'].astype(str)+'_'+\
dnase_positives['start'].astype(str)+'_'+dnase_positives['end'].astype(str)
#dnase_negatives=pd.read_csv('/oak/stanford/groups/akundaje/laks/proseq_chromputer/genome.2kb_h3k27ac_k562_hg19_negatives.bed',sep='\t',header=None)
#dnase_negatives=h3k27ac_negatives[h3k27ac_negatives[1]>20000]
#dnase_negatives.columns=['chr','start','end']
#dnase_negatives=dnase_negatives[dnase_negatives['end']-\
#dnase_negatives['start']==2000]
#dnase_negatives_validation=dnase_negatives[dnase_negatives['chr']=='chr22']
#dnase_negatives['id']=dnase_negatives['chr'].astype(str)+'_'+dnase_negatives['start'].astype(str)+'_'+dnase_negatives['end'].astype(str)
#dnase_negatives=dnase_negatives[dnase_negatives['chr'].isin(['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chrX'])]
#validation_data=dnase_positives_validation.append(dnase_negatives_validation)
validation_data=dnase_positives_validation
test_data=dnase_positives_test
print validation_data[0:5]

input_width=13000 
input_dimension=4
number_of_convolutions=16
filters=[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
filter_dim=[11,11,11,11,11,11,11,11,21,21,21,21,41,41,41,41]
dilation=[1,1,1,1,4,4,4,4,10,10,10,10,25,25,25,25]
activations='relu'
bn_true=True



def BatchNormalization_mod(conv, bn_flag=True):
    from keras.layers.normalization import BatchNormalization
    if bn_flag:
        return BatchNormalization()(conv)
    else:
        return conv


def res_block(conv,num_filter,f_width,act,d_rate,i,bn_true=True):
    import tensorflow as tf
    import keras
    from keras import backend as K
    from keras.layers.pooling import GlobalMaxPooling1D,MaxPooling2D,MaxPooling1D
    from keras.models import Sequential,Model
    from keras.layers import Dense,Activation,Dropout,Flatten,Reshape,Input, Embedding, LSTM, Dense,Concatenate
    from keras.layers.convolutional import Conv1D,Conv2D,Cropping1D
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1,l2
    from keras.optimizers import SGD,RMSprop,Adam
    from sklearn.metrics import average_precision_score
    crop_id=Cropping1D(d_rate*(f_width-1))(conv)
    conv1 = BatchNormalization_mod(conv,bn_true)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv1D(num_filter,f_width,dilation_rate=d_rate,padding="valid",name='conv_'+str(i)+'_a')(conv1)
    conv1 = BatchNormalization_mod(conv1,bn_true)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv1D(num_filter,f_width,dilation_rate=d_rate,padding="valid",name='conv_'+str(i)+'_b')(conv1)
    return keras.layers.Add()([conv1, crop_id])

def build1d_model_residual(input_width,input_dimension,number_of_convolutions,filters,filter_dim,dilation,activations,bn_true=True,max_flag=True):
    import tensorflow as tf
    import keras
    from keras import backend as K
    from keras.layers.pooling import GlobalMaxPooling1D,MaxPooling2D,MaxPooling1D
    from keras.models import Sequential,Model
    from keras.layers import Dense,Activation,Dropout,Flatten,Reshape,Input, Embedding, LSTM, Dense,Concatenate
    from keras.layers.convolutional import Conv1D,Conv2D
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1,l2
    from keras.optimizers import SGD,RMSprop,Adam
    from sklearn.metrics import average_precision_score
    input1=Input(shape=(input_width,4), name='sequence')
    conv=Conv1D(32,1, padding='same',activation='relu',name = 'upsampling')(input1)
    for i in range(0,number_of_convolutions):
            conv = res_block(conv,filters[i],filter_dim[i],activations,dilation[i],i,bn_true)
    conv= Conv1D(32, 1,padding='valid', activation='relu',name='down_sampling')(conv)
    output=Conv1D(1,1,activation='relu',name='dnase')(conv)
    model = Model(input=[input1],output=[output])
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    return model



def validation_loss(model,validation_data):
    validation_mse=[]
    for i in validation_data.values:
            #need to put forward and reverse strand. 
            validation_sequence=fasta_seq[str(i[0])][int(i[1])-5000:int(i[2])+5000]
            validation_sequence=str(validation_sequence).replace('A','0001').replace('C','0010').replace('G','0100').replace('T','1000').replace('N','0000')
            #print len(list(validation_sequence))
            validation_sequence=np.array(list(validation_sequence)).reshape(1,-1,4)
            #seq = Seq(str(c))
            #d=seq.reverse_complement()
            #seq=c
            dnase_values=np.arcsinh(np.nan_to_num(dnase_bigwig_hg38.values(i[0],i[1],i[2])))
            #print dnase_values.shape
            #h3k27ac_value=np.arcsinh(np.nan_to_num(h3k27ac_bigwig_hg38.values(i[0],i[1],i[2])))
            #loss_validation=model.evaluate({'sequence':np.expand_dims(np.asarray(validation_sequence),0) },{'dnase':np.nan_to_num(np.asarray(dnase_values)),'histone':np.nan_to_num(np.asarray(h3k27ac_value))},batch_size=1,verbose=0)
            loss_validation=model.evaluate({'sequence':validation_sequence},{'dnase':np.expand_dims(np.expand_dims(np.asarray(dnase_values),0),2)},batch_size=1,verbose=0)
            validation_mse.append(loss_validation[0])
    return float(sum(validation_mse))/len(validation_mse)




def calculate_input(X_train):
    xtrain=[]
    ytrain_dnase=[]
    ytrain_h3k27ac=[]
    for i in X_train.values:
        j=-1
        while j!=2:
            try:
                start_pos=int(i[1])-(j*1000)
                end_pos=int(i[2])-(j*1000)
                forward_seq=fasta_seq[str(i[0])][start_pos-5000:end_pos+5000]
                forward_seq=str(forward_seq).replace('A','0001').replace('C','0010').replace('G','0100').replace('T','1000').replace('N','0000')
                forward_seq=np.array(list(forward_seq))
                xtrain.append(forward_seq)
                ytrain_dnase.append(np.arcsinh(np.nan_to_num(dnase_bigwig_hg38.values(i[0],start_pos,end_pos))))
                ytrain_h3k27ac.append(np.arcsinh(np.nan_to_num(h3k27ac_bigwig_hg38.values(i[0],start_pos,end_pos))))
                seq = Seq(str(fasta_seq[str(i[0])][start_pos-5000:end_pos+5000]))
                rev_seq=str(seq.reverse_complement()).replace('A','0001').replace('C','0010').replace('G','0100').replace('T','1000').replace('N','0000')
                rev_seq=np.array(list(rev_seq))
                #reverse_seq=str(rev_seq)[::-1]
                xtrain.append(rev_seq)
                ytrain_dnase.append(np.arcsinh(np.nan_to_num(dnase_bigwig_hg38.values(i[0],start_pos,end_pos)))[::-1])
                ytrain_h3k27ac.append(np.arcsinh(np.nan_to_num(h3k27ac_bigwig_hg38.values(i[0],start_pos,end_pos)))[::-1])
                j=j+1
            except Exception as e:
                print str(e)
                print i
    xtrain=np.asarray(xtrain)
    ytrain_dnase=np.expand_dims(np.asarray(ytrain_dnase),2)
    ytrain_h3k27ac=np.expand_dims(np.asarray(ytrain_h3k27ac),2)
    return (xtrain,ytrain_dnase,ytrain_h3k27ac)


model=build1d_model_residual(input_width,input_dimension,number_of_convolutions,filters,filter_dim,dilation,activations,bn_true)
validation_loss_val=10000000
patience=0
while True:
    used_positives=pd.Series()
    used_negatives=pd.Series()
    flag=0
    while True:
        try:
            peaks_regions=dnase_positives[~(dnase_positives['id'].isin(used_positives))].sample(n=150,random_state=95918)
        except:
            used_positives=pd.Series()
            print used_positives.shape
            #print used_negatives.shape
            #flag=flag+1
            #if flag ==100:
            break
            peaks_regions=dnase_positives[~(dnase_positives['id'].isin(used_positives))].sample(n=150,random_state=95918)
            print '#########################breaking#################'
        #try:
        #    nonpeaks_regions=h3k27ac_negatives[~(h3k27ac_negatives['id'].isin(used_negatives))].sample(n=150,random_state=95918)
        #except:
        #    used_negatives=pd.Series()
        used_positives=used_positives.append(peaks_regions['id'])
        #used_negatives=used_negatives.append(nonpeaks_regions['id'])
        #X_train=peaks_regions.append(nonpeaks_regions)
        print used_positives.shape
        X_train=peaks_regions
        X_train=X_train.sample(frac=1)
        X_train_sequence,Y_train_dnase,Y_train_histone=calculate_input(X_train)
        X_train_sequence=X_train_sequence.reshape(X_train_sequence.shape[0],-1,4)
        a=model.fit({'sequence': X_train_sequence },{'dnase':Y_train_dnase},epochs=1, batch_size=60,verbose=0)    
    new_validation_loss_val=validation_loss(model,validation_data)
    if new_validation_loss_val < validation_loss_val:
        print '#############change in value ##############'
        print new_validation_loss_val
        print validation_loss_val
        print '#############change in value ##############'
        validation_loss_val=new_validation_loss_val
        model.save_weights('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/13kb_context_3b_prediction_dnase_Fwd_rev_squiggle.hdf5')
    else:
        print '############# no change in value ##############'
        print new_validation_loss_val
        print validation_loss_val
        print '############# no change in value ##############'
        model.load_weights('/oak/stanford/groups/akundaje/laks/k562_expression_accessibility_histone/13kb_context_3b_prediction_dnase_Fwd_rev_squiggle.hdf5')
        patience=patience+1
    if patience==5:
        break



