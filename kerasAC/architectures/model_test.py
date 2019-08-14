from profile_model import *
import argparse
import pdb

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",default=1234)
    parser.add_argument("--init_weights",default=None)
    parser.add_argument("--sequence_flank",default=6500)
    return parser.parse_args()

def main():
    args=parse_args()
    model=getModelGivenModelOptionsAndWeightInits(args)
    print(model.summary())
    
if __name__ == "__main__":
    main()
    
