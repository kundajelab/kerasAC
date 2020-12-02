import argparse
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--file_name_prefix")
    parser.add_argument("--file_name_suffix")
    parser.add_argument("--outf")
    parser.add_argument("--nsplits",type=int,default=10)
    return parser.parse_args()

def main():
    args=parse_args()
    aggregate_dict={}
    tasks=None
    for split in range(args.nsplits):
        try:
            aggregate_dict[split]=dict()
            fname=args.file_name_prefix+str(split)+args.file_name_suffix
            print(fname)
            data=open(fname,'r').read().strip().split('\n')
            if tasks is None:
                header=data[0].split('\t')
                tasks=header
                print(str(tasks))
            for line in data[1::]:
                tokens=line.split('\t')
                metric=tokens[0]
                aggregate_dict[split][metric]={} 
                for i in range(len(tasks)):
                    cur_task=tasks[i] 
                    cur_val=round(float(tokens[i+1].split('(')[-1].split(')')[0].split(',')[0].split('=')[-1]),3)
                    #print(cur_val)
                    aggregate_dict[split][metric][cur_task]=cur_val
        except:
            print("skipping split:"+str(split))
    print(aggregate_dict)
    outf=open(args.outf,'w')
    outf.write('Split\tMetrics\t'+'\t'.join(tasks)+'\n')
    for split in range(args.nsplits):
        for metric in aggregate_dict[split]:
            try:
                outf.write(str(split)+'\t'+str(metric))
                for task in tasks:
                    outf.write('\t'+str(aggregate_dict[split][metric][task]))
                outf.write('\n')
            except:
                continue 
                
if __name__=="__main__":
    main()
    
#prefix="gecco.200bpregression.with.gc.performance.fold"
#suffix=".tsv"
