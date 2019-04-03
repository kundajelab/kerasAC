from kerasAC.custom_losses import *
from keras.models import load_model
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1
    
custom_objects={"recall":recall,
                "sensitivity":recall,
                "specificity":specificity,
                "fpr":fpr,
                "fnr":fnr,
                "precision":precision,
                "f1":f1,
                "ambig_binary_crossentropy":get_ambig_binary_crossentropy(),
                "ambig_mean_squared_error":get_ambig_mean_squared_error()}
model_hdf5="/srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/classification_init_dan_model/DNASE.K562.classification.SummitWithin200bpCenter.0"
model=load_model(model_hdf5,custom_objects=custom_objects)
