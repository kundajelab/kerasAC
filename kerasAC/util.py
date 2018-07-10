import sys, os
import os.path
from collections import OrderedDict



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


