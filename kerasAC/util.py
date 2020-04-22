import sys, os
import os.path
from collections import OrderedDict
import argparse
import numpy as np 
import tiledb

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

def one_hot_encode(seqs):
    return np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])

def dinuc_shuffle(seq):
    #get list of dinucleotides
    nucs=[]
    for i in range(0,len(seq),2):
        nucs.append(seq[i:i+2])
    #generate a random permutation
    random.shuffle(nucs)
    return ''.join(nucs) 


def revcomp(seq):
    seq=seq[::-1].upper()
    comp_dict=dict()
    comp_dict['A']='T'
    comp_dict['T']='A'
    comp_dict['C']='G'
    comp_dict['G']='C'
    rc=[]
    for base in seq:
        if base in comp_dict:
            rc.append(comp_dict[base])
        else:
            rc.append(base)
    return ''.join(rc)






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


    
    
def coords_to_tdb_indices(coords,tdb_instance):
    '''
    coords is a tuple (chrom, start, stop)
    '''
    num_chroms=tdb_instance.meta['num_chroms']
    for i in range(num_chroms):
        if tdb_instance.meta['chrom_'+str(i)]==coords[0]:
            chrom_offset=tdb_instance.meta['offset_'+str(i)]
            tdb_index_start=chrom_offset+coords[1]
            tdb_index_end=chrom_offset+coords[2]
            return (tdb_index_start,tdb_index_end)
    raise Exception("chrom name:"+str(coords[0])+" not found in tdb array")


def tdb_indices_to_coords(indices,tdb_instance):
    '''
    indices is a list of tdb indices     
    '''
    pass


def transform_data_type(inputs,num_inputs):
    if inputs is None:
        inputs=[None]*num_inputs
    elif inputs is []:
        inputs=[None]*num_inputs
    else:
        assert(len(inputs)==num_inputs)
        for i in range(num_inputs):
            if str(inputs[i]).lower()=="none":
                inputs[i]=None
            else:
                inputs[i]=float(inputs[i])
    return inputs 
                
