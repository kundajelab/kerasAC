import random
import tensorflow
from tensorflow.compat.v1.keras.backend import get_session
tensorflow.compat.v1.disable_v2_behavior()
import math
import kerasAC 
from scipy.special import softmax,expit
from kerasAC.interpret.deepshap import * 
from kerasAC.interpret.profile_shap import * 
from kerasAC.helpers.transform_bpnet_io import * 
from kerasAC.util import * 
import pandas as pd


def get_ism_scrambled_region(model,seq,prof_pred,count_pred,mask_size=5): 
    #expand default preds to match dimensions for ISM 
    
    scrambled_seq=''.join(random.sample(seq,len(seq)))
    default_prof_expanded=np.zeros((len(seq),1000))
    default_count_expanded=np.zeros((len(seq)))
    for i in range(len(seq)): 
        default_prof_expanded[i,:]=prof_pred 
        default_count_expanded[i]=count_pred
            
    #create placeholders for ISM predictions        
    placeholder_prof=np.zeros((len(seq),1000))
    placeholder_count=np.zeros((len(seq)))

    for i in range(len(seq)): 
        mask_start=max([0,i-int(mask_size)//2])
        mask_end=min([len(seq),i+1+int(mask_size)//2])
        mask_length=mask_end-mask_start

        cur_seq=seq[0:mask_start]+scrambled_seq[mask_start:mask_end]+seq[mask_end::]
        #get predictions for the sequence with 0 mask  
        preds=model.predict([one_hot_encode([cur_seq])])
        prof=np.squeeze(preds[0])
        count=np.squeeze(preds[1])
        placeholder_prof[i,:]=prof
        placeholder_count[i]=count
    
    #subtract the WT, average across base axis 
    placeholder_prof_normed=-1*(placeholder_prof-default_prof_expanded)
    placeholder_count_normed=-1*(placeholder_count-default_count_expanded)
   
    seq_onehot=np.squeeze(one_hot_encode([seq]))
    ism_count_track=np.expand_dims(placeholder_count_normed,axis=1)*seq_onehot
    return np.squeeze(ism_count_track), np.squeeze(placeholder_prof_normed)


def get_ism_single_bp(model,seq,prof_pred,count_pred): 
    #expand default preds to match dimensions for ISM 
    default_prof_expanded=np.zeros((len(seq),1000,4))
    default_count_expanded=np.zeros((len(seq),4))
    for j in range(4): 
        for i in range(len(seq)): 
            default_prof_expanded[i,:,j]=prof_pred 
            default_count_expanded[i,j]=count_pred
            
    #create placeholders for ISM predictions        
    ind_to_base={0:'A',1:'C',2:'G',3:'T'}
    placeholder_prof=np.zeros((len(seq),1000,4))
    placeholder_count=np.zeros((len(seq),4))

    for j in range(4):
        cur_allele_seqs=[]
        for i in range(len(seq)): 
            cur_allele_seqs.append(seq[0:i]+ind_to_base[j]+seq[i+1::])
        #get predictions for this allele 
        cur_allele_preds=model.predict([one_hot_encode(cur_allele_seqs)])
        cur_allele_prof=np.squeeze(cur_allele_preds[0])
        cur_allele_count=np.squeeze(cur_allele_preds[1])
        placeholder_prof[:,:,j]=cur_allele_prof
        placeholder_count[:,j]=cur_allele_count
    
    #subtract the WT, average across base axis 
    placeholder_prof_normed=placeholder_prof-default_prof_expanded
    placeholder_count_normed=placeholder_count-default_count_expanded

    placeholder_prof_normed=placeholder_prof_normed-np.expand_dims(np.mean(placeholder_prof_normed,axis=2),axis=2)
    placeholder_count_normed=placeholder_count_normed-np.expand_dims(np.mean(placeholder_count_normed,axis=1),axis=1)
    
    seq_onehot=one_hot_encode([seq])
    ism_count_track=placeholder_count_normed*seq_onehot
    
    #observed base heatmap
    ism_mat_observed=np.sum(np.expand_dims(np.squeeze(seq_onehot),axis=1)*placeholder_prof_normed,axis=2)
    
    return  np.squeeze(ism_count_track), np.squeeze(ism_mat_observed)

def analyze_background(ref,
            chrom,
            summit,
            ref_allele,
            alt_allele,
            rsid,
            model,
            flank=673):
    #get the reference and alternate one-hot-encoded sequences 
    seq=ref.fetch(chrom,summit-flank,summit+flank)
    #SCRAMBLE!! 
    seq=''.join(random.sample(seq,len(seq)))
    
    ref_seq=seq[0:flank]+ref_allele+seq[flank+1::]
    assert len(ref_seq)==2*flank
    ref_onehot=one_hot_encode([ref_seq])
    
    alt_seq=seq[0:flank]+alt_allele+seq[flank+1::]
    assert len(alt_seq)==2*flank
    alt_onehot=one_hot_encode([alt_seq])
    
        
    #get predictions for reference & alternate allele 
    prof_ref,count_ref,probs_ref,count_track_ref=get_preds(model,ref_onehot)
    prof_alt,count_alt,probs_alt,count_track_alt=get_preds(model,alt_onehot)

    #get ISM scores 
    single_bp_count_flat, single_bp_ism_mat_observed_flat=get_ism_single_bp_sub(model,seq,prof_ref,count_ref)
    #get masked ISM scores (scrambled)
    scrambled_region_count_flat, scrambled_region_ism_mat_flat =get_ism_scrambled_region(model,seq,prof_ref,count_ref)
    return single_bp_ism_mat_observed_flat, single_bp_count_flat, scrambled_region_ism_mat_flat,scrambled_region_count_flat


def filter_ism_by_pval(ism,background,isprof,pthresh=0.05): 
    print("ism.shape:"+str(ism.shape))
    print("sum start:"+str(ism.sum()))
    ism_pvals=1-background(abs(ism))
    mask=np.asarray(ism_pvals<=pthresh).astype('int')
    ism=ism*mask 
    print(pthresh)
    print("sum end:"+str(ism.sum()))
    if isprof==False:
        return ism 
    mask=np.sum(ism,axis=1) #2114x4
    mask[mask>0]=1
    mask[mask<0]=-1
    return np.sum(abs(ism),axis=1)*mask #2114x4


def analyze(ref,
            chrom,
            summit,
            ref_allele,
            alt_allele,
            rsid,
            bigwig,
            model,
            count_explainer,
            prof_explainer,
            background_single_bp_prof,
            background_single_bp_count,
            background_scrambled_prof,
            background_scrambled_count,
            flank=673):
    #get the reference and alternate one-hot-encoded sequences 
    seq=ref.fetch(chrom,summit-flank,summit+flank)
    
    ref_seq=seq[0:flank]+ref_allele+seq[flank+1::]
    assert len(ref_seq)==2*flank
    ref_onehot=one_hot_encode([ref_seq])
    
    alt_seq=seq[0:flank]+alt_allele+seq[flank+1::]
    assert len(alt_seq)==2*flank
    alt_onehot=one_hot_encode([alt_seq])
    
    #get the bigwig labels 
    labels=np.nan_to_num(bigwig.values(chrom,summit-flank,summit+flank))
    
    #get predictions for reference & alternate allele 
    prof_ref,count_ref,probs_ref,count_track_ref=get_preds(model,ref_onehot)
    prof_alt,count_alt,probs_alt,count_track_alt=get_preds(model,alt_onehot)
    
    #get the log odds blast radius track 
    blast_radius_track=[0]*557+(np.log(probs_ref)-np.log(probs_alt)).tolist()+[0]*557
    #get deepSHAP scores for ref & alt alleles 
    profile_explanations_ref, count_explanations_ref=get_deepshap(prof_explainer, count_explainer, ref_onehot)
    profile_explanations_alt, count_explanations_alt=get_deepshap(prof_explainer, count_explainer, alt_onehot)
    
    #get ISM scores 
    single_bp_ism_profile_track, \
    single_bp_ism_count_track, \
    single_bp_ism_profile_track_filtered_p_sign, \
    single_bp_ism_count_track_filtered_p_sign, \
    single_bp_ism_profile_track_filtered_sign, \
    single_bp_ism_count_track_filtered_sign, \
    single_bp_ism_mat_observed=get_ism_single_bp_sub(model,
                                                     seq,
                                                     prof_ref,
                                                     count_ref,
                                                     background_single_bp_prof,
                                                     background_single_bp_count)
        
    #get masked ISM scores (scrambled)
    scrambled_region_ism_profile_track, \
    scrambled_region_ism_count_track, \
    scrambled_region_ism_profile_track_filtered_p_sign, \
    scrambled_region_ism_count_track_filtered_p_sign, \
    scrambled_region_ism_profile_track_filtered_sign, \
    scrambled_region_ism_count_track_filtered_sign, \
    scrambled_region_ism_mat =get_ism_scrambled_region(model,
                                                       seq,
                                                       prof_ref,
                                                       count_ref,
                                                       background_scrambled_prof,
                                                       background_scrambled_count)
    import pdb 
    #visualize 
    make_plot(labels,
              [0]*557+count_track_ref.tolist()+[0]*557,
              [0]*557+count_track_alt.tolist()+[0]*557,
              count_ref,
              count_alt,
              blast_radius_track,
              profile_explanations_ref,
              count_explanations_ref,
              profile_explanations_alt,
              count_explanations_alt,
              single_bp_ism_profile_track,
              single_bp_ism_profile_track_filtered_p_sign,
              single_bp_ism_profile_track_filtered_sign,
              single_bp_ism_count_track,
              single_bp_ism_count_track_filtered_p_sign,
              single_bp_ism_count_track_filtered_sign,
              scrambled_region_ism_profile_track,
              scrambled_region_ism_profile_track_filtered_p_sign,              
              scrambled_region_ism_profile_track_filtered_sign,              
              scrambled_region_ism_count_track,
              scrambled_region_ism_count_track_filtered_p_sign,
              scrambled_region_ism_count_track_filtered_sign,              
              single_bp_ism_mat_observed,
              scrambled_region_ism_mat,
              ':'.join([str(chrom),str(summit),str(ref_allele),str(alt_allele),str(rsid)]))

