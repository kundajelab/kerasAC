import argparse
import pandas as pd
from os import listdir
from os.path import isfile, join

def parse_args():
    parser=argparse.ArgumentParser(description="aggregate MEME files for MODISCO patterns with significant TomTom hits")
    parser.add_argument("--tomtom_results")
    parser.add_argument("--modisco_meme_file_dir")
    parser.add_argument("--outf")
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
        contents=open(join(args.modisco_meme_file_dir,meme_file),'r').read()
        first_to_keep=contents.find('letter-probability matrix')
        meme_file_dict[pattern_name]=contents[first_to_keep::]
        
    outf=open(args.outf,'w')
    #write the normal meme header stuff
    outf.write('MEME version 4\n\nALPHABET= ACGT\n\nstrands:  + -\n\nBackground letter frequencies (from unknown source):\nA 0.250 C 0.250 G 0.250 T 0.250\n\n')

    #aggregate motifs that are tomtom hits
    for index,row in tomtom_results.iterrows():
        pattern=row['Pattern']
        header='|'.join([str(i) for i in row])
        outf.write('MOTIF '+header+'\n'+meme_file_dict[pattern]+'URL none\n\n')
    outf.close()
    

if __name__=='__main__':
    main()
    
