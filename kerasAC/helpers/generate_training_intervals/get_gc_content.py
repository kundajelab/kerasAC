import pandas as pd
import pysam
import argparse
import pickle
def parse_args():
    parser=argparse.ArgumentParser(description="get gc content from a bed file")
    parser.add_argument("--input_bed")
    parser.add_argument("--ref_fasta")
    parser.add_argument("--out_pickle")
    return parser.parse_args()

def main():
    args=parse_args()
    ref=pysam.FastaFile(args.ref_fasta)
    outputs=dict()
    outf=None
    data=pd.read_csv(args.input_bed,header=None,sep='\t')    
    print("loaded bed file")
    num_rows=str(data.shape[0])
    print("num_rows:"+num_rows)
    for index,row in data.iterrows():
        if index%1000==0:
            print(str(index))
        seq=ref.fetch(row[0],row[1],row[2]).upper() 
        g_count=seq.count('G')
        c_count=seq.count('C')
        gc_fract=round((g_count+c_count)/len(seq),2)
        if gc_fract not in outputs:
            outputs[gc_fract]=['\t'.join([str(i) for i in row])]
        else:
            outputs[gc_fract].append('\t'.join([str(i) for i in row]))
    print("pickling") 
    with open(args.out_pickle,'wb') as handle:
        pickle.dump(outputs,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print("done!")
    
if __name__=="__main__":
    main()
