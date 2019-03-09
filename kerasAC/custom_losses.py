import keras.backend as K
import tensorflow as tf
import pdb 
import numpy as np 
def get_weighted_binary_crossentropy(w0_weights, w1_weights,ambig_val=np.nan):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    import numpy as np 
    w0_weights=np.array(w0_weights);
    w1_weights=np.array(w1_weights);
    def weighted_binary_crossentropy(y_true,y_pred):
        weightsPerTaskRep = y_true*w1_weights[None,:] + (1-y_true)*w0_weights[None,:]
        nonAmbig=tf.math.logical_not(tf.is_nan(y_true))
        nonAmbigTimesWeightsPerTask = tf.boolean_mask(weightsPerTaskRep,nonAmbig)
        return K.mean(K.binary_crossentropy(tf.boolean_mask(y_true,nonAmbig),tf.boolean_mask(y_pred,nonAmbig))*nonAmbigTimesWeightsPerTask, axis=-1);
    return weighted_binary_crossentropy; 

def get_ambig_binary_crossentropy():
    def ambig_binary_crossentropy(y_true,y_pred):
        nonAmbig=tf.math.logical_not(tf.is_nan(y_true))
        return K.mean(K.binary_crossentropy(tf.boolean_mask(y_true,nonAmbig), tf.boolean_mask(y_pred,nonAmbig)), axis=-1);
    return ambig_binary_crossentropy; 

def get_ambig_mean_squared_error(ambig_val=np.nan): 
    def ambig_mean_squared_error(y_true, y_pred):
        nonAmbig=tf.math.logical_not(tf.is_nan(y_true))
        return K.mean(K.square(tf.boolean_mask(y_pred,nonAmbig) - tf.boolean_mask(y_true,nonAmbig)), axis=-1)
    return ambig_mean_squared_error
