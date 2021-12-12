import argparse
import pandas as pd 
import pdb
import os
pd.options.display.max_colwidth = 500

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--motif_dir',help='directory where motif logo png images are stored')
    parser.add_argument('--tomtom_hits_file')
    parser.add_argument('--modisco_plots_folder')
    parser.add_argument("--html_prefix_motif_ref_pngs")
    parser.add_argument("--html_prefix_modisco_pngs") 
    parser.add_argument("--outf")
    parser.add_argument("--max_tomtom_hits",type=int,default=5) 
    return parser.parse_args()


def path_to_image_html(path):
    return '<img src="'+ path + '" width="240" >'
    

def main():
    args=parse_args()
    tomtom_hits=pd.read_csv(args.tomtom_hits_file,header=0,sep='\t')
    tomtom_hits['modisco_cwm_fwd']=None
    tomtom_hits['modisco_cwm_rev']=None
    for i in range(args.max_tomtom_hits):
        tomtom_hits['match_'+str(i+1)+'_logo']=None
    for index,row in tomtom_hits.iterrows():
        
        cur_pattern=row['Pattern']
        #get the forward cwm if it exists
        forward_cwm_pattern_file='/'.join([args.modisco_plots_folder,'metacluster_1.'+cur_pattern+'.cwm.fwd.png'])
        #get the reverse cwm if it exists
        reverse_cwm_pattern_file='/'.join([args.modisco_plots_folder,'metacluster_1.'+cur_pattern+'.cwm.rev.png'])
        if os.path.exists(forward_cwm_pattern_file):
            #store in df
            tomtom_hits['modisco_cwm_fwd'][index]='/'.join([args.html_prefix_modisco_pngs,'metacluster_1.'+cur_pattern+'.cwm.fwd.png'])
        if os.path.exists(reverse_cwm_pattern_file):
            #store in df
            tomtom_hits['modisco_cwm_rev'][index]='/'.join([args.html_prefix_modisco_pngs,'metacluster_1.'+cur_pattern+'.cwm.rev.png'])
        for i in range(args.max_tomtom_hits):
            #check if we have a hit and fetch the png if we do
            if pd.isna(row['match_'+str(i+1)]) is False: 
                tomtom_hits['match_'+str(i+1)+'_logo'][index]='/'.join([args.html_prefix_motif_ref_pngs,row['match_'+str(i+1)]+'.pfm.png'])
    #order the columns
    columns=['Pattern','num_seqlets','modisco_cwm_fwd','modisco_cwm_rev']
    formatters={'modisco_cwm_fwd':path_to_image_html,
                'modisco_cwm_rev':path_to_image_html} 
    for i in range(args.max_tomtom_hits):
        columns.append('match_'+str(i+1))
        columns.append('qvalue_'+str(i+1))
        columns.append('match_'+str(i+1)+'_logo')
        formatters['match_'+str(i+1)+'_logo']=path_to_image_html
        
    tomtom_hits=tomtom_hits.reindex(columns=columns)
    tomtom_hits.to_html(args.outf,
                        escape=False,
                        formatters=formatters,
                        index=False)
    
    
    

if __name__=="__main__":
    main()
    
