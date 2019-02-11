import sys, os
import os.path
from collections import OrderedDict
import argparse



ltrdict = {'a':[1,0,0,0],
           'c':[0,1,0,0],
           'g':[0,0,1,0],
           't':[0,0,0,1],
           'n':[0,0,0,0],
           'A':[1,0,0,0],
           'C':[0,1,0,0],
           'G':[0,0,1,0],
           'T':[0,0,0,1],
           'N':[0,0,0,0]}



class DefaultOrderedDictWrapper(object):
    def __init__(self, factory):
        self.ordered_dict = OrderedDict()
        assert hasattr(factory, '__call__')
        self.factory = factory

    def __getitem__(self, key):
        if key not in self.ordered_dict:
            self.ordered_dict[key] = self.factory() 
        return self.ordered_dict[key]



def enum(**enums):
    class Enum(object):
        pass
    to_return = Enum
    for key,val in enums.items():
        if hasattr(val, '__call__'): 
            setattr(to_return, key, staticmethod(val))
        else:
            setattr(to_return, key, val)
    to_return.vals = [x for x in enums.values()]
    to_return.the_dict = enums
    return to_return


def combine_enums(*enums):
    new_enum_dict = OrderedDict()
    for an_enum in enums:
        new_enum_dict.update(an_enum.the_dict)
    return enum(**new_enum_dict)



def args_object_from_args_dict(args_dict):
    #create an argparse.Namespace from the dictionary of inputs
    args_object=argparse.Namespace()
    #set the defaults for training/prediction/interpretation/cross-validation
    #training 
    vars(args_object)['batch_size']=1000
    vars(args_object)['num_train']=700000
    vars(args_object)['num_valid']=150000
    vars(args_object)['ref_fasta']='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa'
    vars(args_object)['init_weights']=None
    vars(args_object)['w0_file']=None
    vars(args_object)['w1_file']=None
    vars(args_object)['from_checkpoint_weights']=None
    vars(args_object)['from_checkpoint_arch']=None
    vars(args_object)['vcf_file']=None
    vars(args_object)['epochs']=40
    #prediction

    #cross-validation

    
    for key in args_dict:
        vars(args_object)[key]=args_dict[key]
        
    
    
