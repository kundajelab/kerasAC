import keras.backend as K
def get_weighted_binary_crossentropy(w0_weights, w1_weights):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    import numpy as np 
    w0_weights=np.array(w0_weights);
    w1_weights=np.array(w1_weights);
    thresh=-0.5
    def weighted_binary_crossentropy(y_true,y_pred):
        weightsPerTaskRep = y_true*w1_weights[None,:] + (1-y_true)*w0_weights[None,:]
        nonAmbig = K.cast((y_true > -0.5),'float32')
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_pred, y_true)*nonAmbigTimesWeightsPerTask, axis=-1);
    return weighted_binary_crossentropy; 


