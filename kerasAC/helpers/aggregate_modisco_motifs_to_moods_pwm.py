import argparse
import numpy as np 
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists

def parse_args():
    parser=argparse.ArgumentParser(description="aggregate MEME files for MODISCO patterns with significant TomTom hits")
    parser.add_argument("--tomtom_results")
    parser.add_argument("--modisco_meme_file_dir")
    parser.add_argument("--out_dir")
    return parser.parse_args()

def main():
    args=parse_args()
    tomtom_results=pd.read_csv(args.tomtom_results,header=0,sep='\t')
    #remove all the ones where there were no hits
    tomtom_results=tomtom_results.dropna(subset=['match_1'])
    
    #get all the files in the modisco meme output dir
    meme_files=[f for f in listdir(args.modisco_meme_file_dir) if isfile(join(args.modisco_meme_file_dir,f))]
    meme_file_dict={}
    for meme_file in meme_files:
        pattern_name=meme_file.split('.')[-2]
        contents=np.transpose(np.loadtxt(join(args.modisco_meme_file_dir,meme_file),skiprows=12))
        contents[contents==0]=1e-6
        meme_file_dict[pattern_name]=contents

    #create output dir if it doesn't exist
    if not exists(args.out_dir):
        makedirs(args.out_dir)
        
    #aggregate motifs that are tomtom hits
    for index,row in tomtom_results.iterrows():
        pattern=row['Pattern']
        np.savetxt(join(args.out_dir,pattern),meme_file_dict[pattern],delimiter='\t')
    

if __name__=='__main__':
    main()
    
