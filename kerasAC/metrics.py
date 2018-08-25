#Note: this is ugly w/ use of tf & K --> needed to avoid custom keras modifications 
import tensorflow as tf
import keras.backend as K
def spearman_corr(y_true,y_pred):
    import K.contribs.metrics.streaming_pearson_correlation
    return K.contribs.metrics.streaming_pearson_correlation(y_pred,y_true)

def positive_accuracy(y_true,y_pred):
    one_indices=tf.cast(tf.where(tf.equal(y_true,1.0)),'int32')
    y_true_subset=tf.gather_nd(y_true,one_indices)
    y_pred_subset=tf.gather_nd(y_pred,one_indices)
    positive_accuracy=K.mean(tf.equal(y_true_subset,tf.round(y_pred_subset)))
    return positive_accuracy


def negative_accuracy(y_true,y_pred):
    one_indices=tf.cast(tf.where(tf.equal(y_true,0.0)),'int32')
    y_true_subset=tf.gather_nd(y_true,one_indices)
    y_pred_subset=tf.gather_nd(y_pred,one_indices)
    negative_accuracy=K.mean(tf.equal(y_true_subset,tf.round(y_pred_subset)))
    return negative_accuracy
                                        


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(tf.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives =K.sum(tf.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(tf.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(tf.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
                                        



