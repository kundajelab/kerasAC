import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp

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
        nonAmbig=tf.math.logical_not(tf.math.is_nan(y_true))
        nonAmbigTimesWeightsPerTask = tf.boolean_mask(weightsPerTaskRep,nonAmbig)
        return K.mean(K.binary_crossentropy(tf.boolean_mask(y_true,nonAmbig),tf.boolean_mask(y_pred,nonAmbig))*nonAmbigTimesWeightsPerTask, axis=-1);
    return weighted_binary_crossentropy; 

def ambig_binary_crossentropy(y_true,y_pred):
    nonAmbig=tf.math.logical_not(tf.math.is_nan(y_true))
    return K.mean(K.binary_crossentropy(tf.boolean_mask(y_true,nonAmbig), tf.boolean_mask(y_pred,nonAmbig)), axis=-1);

def ambig_mean_squared_error(y_true, y_pred):
    nonAmbig=tf.math.logical_not(tf.math.is_nan(y_true))
    return K.mean(K.square(tf.boolean_mask(y_pred,nonAmbig) - tf.boolean_mask(y_true,nonAmbig)), axis=-1)

def ambig_mean_absolute_error(y_true, y_pred):
    nonAmbig=tf.math.logical_not(tf.math.is_nan(y_true))
    return K.mean(K.abs(tf.boolean_mask(y_pred,nonAmbig) - tf.boolean_mask(y_true,nonAmbig)), axis=-1)

def ambig_log_poisson(y_true,y_pred):
    nonAmbig=tf.math.logical_not(tf.math.is_nan(y_true))
    return tf.nn.log_poisson_loss(nonAmbig,tf.log(y_pred),compute_full_loss=True)


#PROFILE MODEL LOSSES #
def get_loss_weights(tdb_path,chrom,label_attribute,ambig_attribute,upsample_attribute,tdb_partition_thresh_for_upsample):
    import tiledb
    from kerasAC.tiledb_config import get_default_config
    import pdb 
    tdb_config=get_default_config()
    ctx=tiledb.Ctx(tdb_config)
    tdb_array=tiledb.DenseArray(tdb_path+"."+chrom,mode='r',ctx=ctx)
    print("opened:"+tdb_path+"."+chrom+" for reading")
    vals=tdb_array[:]
    print("got tdb vals")
    #label_attribute

def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))

#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
class MultichannelMultinomialNLL(object):
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}

