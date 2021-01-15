from scipy.special import softmax
import matplotlib 
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"]=10,5
plt.rcParams['axes.xmargin'] = 0

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

def make_plot(label_counts_track,
              ref_pred_count_track,
              alt_pred_count_track,
              ref_pred_count_val,
              alt_pred_count_val,
              log_odds_prob,
              deepshap_prof_ref,
              deepshap_count_ref,
              deepshap_prof_alt,
              deepshap_count_alt,
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
              title,
              xmin=0,
              xmax=2114): 
    plt.rcParams["figure.figsize"]=15,28
    f,axes=plt.subplots(21, 1,sharex='row')
    plt.set_cmap('RdBu')
    
    #labels count track 
    axes[0].plot(label_counts_track)  
    axes[0].set_title("Label Counts")
    
    #alt & ref predicted count tracks 
    axes[1].plot(ref_pred_count_track,label='Ref pred counts per base',color='b')
    axes[1].plot(alt_pred_count_track,label='Alt pred counts per base',color='r')
    axes[1].set_title("Counts Pred Ref:"+str(ref_pred_count_val)+":"+"Counts Pred Alt:"+str(alt_pred_count_val))
    axes[1].legend() 
    
    #log-ratio predicted probs (blast radius track)
    axes[2].plot(log_odds_prob)
    axes[2].set_title("log(prediced prob(ref))-log(predicted prob(alt))")
    
    #plot DeepSHAP tracks for profile & counts, for ref & alt alleles 
    axes[3]=plot_bases_on_ax(deepshap_prof_ref,axes[3],show_ticks=False)
    axes[3].set_title('DeepSHAP Profile Ref')
    axes[3].set_ylim(-0.05,0.05)
    
    axes[4]=plot_bases_on_ax(deepshap_prof_alt,axes[4],show_ticks=False)
    axes[4].set_title('DeepSHAP Profile Alt')
    axes[4].set_ylim(-0.05,0.05)
    
    axes[5]=plot_bases_on_ax(deepshap_count_ref,axes[5],show_ticks=False)
    axes[5].set_title('DeepSHAP Count Ref')
    axes[5].set_ylim(-0.05,0.05)

    axes[6]=plot_bases_on_ax(deepshap_count_alt,axes[6],show_ticks=False)
    axes[6].set_title('DeepSHAP Count Alt')
    axes[6].set_ylim(-0.05,0.05)

    axes[7]=plot_bases_on_ax(single_bp_ism_profile_track,axes[7],show_ticks=False)
    axes[7].set_title("ISM Profile, single bp")

    axes[8]=plot_bases_on_ax(single_bp_ism_profile_track_filtered_sign, axes[8],show_ticks=False)
    axes[8].set_title("ISM Profile, single bp, filtered by sign")
    
    axes[9]=plot_bases_on_ax(single_bp_ism_profile_track_filtered_p_sign, axes[9],show_ticks=False)
    axes[9].set_title("ISM Profile, single bp, filtered by sign, p-value<0.05")
 
    axes[10]=plot_bases_on_ax(single_bp_ism_count_track,axes[10],show_ticks=False)
    axes[10].set_title("ISM Count, single bp")
    
    axes[11]=plot_bases_on_ax(single_bp_ism_count_track_filtered_sign,axes[11],show_ticks=False)
    axes[11].set_title("ISM Count, single bp, filtered by sign")

    axes[12]=plot_bases_on_ax(single_bp_ism_count_track_filtered_p_sign,axes[12],show_ticks=False)
    axes[12].set_title("ISM Count, single bp, filtered by sign, p-value <0.05")
         
    axes[13]=plot_bases_on_ax(scrambled_region_ism_profile_track,axes[13],show_ticks=False)
    axes[13].set_title("ISM Profile, 5bp, scrambled region")
    
    axes[14]=plot_bases_on_ax(scrambled_region_ism_profile_track_filtered_sign,axes[14],show_ticks=False)
    axes[14].set_title("ISM Profile, 5bp, scrambled region, filtered by sign")

    axes[15]=plot_bases_on_ax(scrambled_region_ism_profile_track_filtered_p_sign,axes[15],show_ticks=False)
    axes[15].set_title("ISM Profile, 5bp, scrambled region, filtered by sign, p-value < 0.05")
       
    axes[16]=plot_bases_on_ax(scrambled_region_ism_count_track,axes[16],show_ticks=False)
    axes[16].set_title("ISM Count, 5bp, scrambled region")

    axes[17]=plot_bases_on_ax(scrambled_region_ism_count_track_filtered_sign,axes[17],show_ticks=False)
    axes[17].set_title("ISM Count, 5bp, scrambled region, filtered_by_sign")
    
    axes[18]=plot_bases_on_ax(scrambled_region_ism_count_track_filtered_p_sign,axes[18],show_ticks=False)
    axes[18].set_title("ISM Count, 5bp, scrambled region, filtered_by_sign, p-value. < 0.05")
    
    for i in range(19): 
        axes[i].set_xlim(xmin,xmax)
        axes[i].set_xticks(list(range(xmin, xmax, 100)))  
        
    extent = [0, single_bp_ism_mat_observed.shape[0], 0, single_bp_ism_mat_observed.shape[1]]
    ymin=min([np.amin(scrambled_region_ism_mat),np.amin(single_bp_ism_mat_observed)])
    ymax=max([np.amax(scrambled_region_ism_mat),np.amax(single_bp_ism_mat_observed)])
    #abs_highest=max([abs(ymin),abs(ymax)])
    hmap1=axes[19].imshow(single_bp_ism_mat_observed.T,
                        extent=extent,
                        vmin=-0.5,
                        vmax=0.5, 
                        interpolation='nearest',aspect='auto')
    axes[19].set_title("ISM matrix, single bp, observed allele projected")
    
    hmap2=axes[20].imshow(scrambled_region_ism_mat.T,
                        extent=extent,
                        vmin=-0.5,
                        vmax=0.5, 
                        interpolation='nearest',aspect='auto')
    axes[20].set_title("ISM matrix, 5bp scrambled")
    
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.colorbar(hmap1,ax=axes[19],orientation='horizontal')
    plt.colorbar(hmap2,ax=axes[20],orientation='horizontal')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(title.replace('/',':')+'.png',format='png',dpi=300)
    plt.show()
