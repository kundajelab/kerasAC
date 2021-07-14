#functional factorized bassett? 
import numpy as np ;
from tensorflow.keras.constraints import max_norm
from kerasAC.metrics import *
from kerasAC.custom_losses import get_weighted_binary_crossentropy, get_ambig_binary_crossentropy
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1

def getModelGivenModelOptionsAndWeightInits(args):
    #read in the arguments
    w0=args.w0
    w1=args.w1
    init_weights=args.init_weights
    seed=args.seed
    
    np.random.seed(seed)
    import keras;
    from tensorflow.keras.layers import (
        Activation, AveragePooling1D, BatchNormalization,
        Conv1D, Conv2D, Dense, Dropout, Flatten, Input,
        MaxPooling1D, MaxPooling2D, Reshape,
        PReLU, Add
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adadelta, SGD, RMSprop;
    import keras.losses;
    from tensorflow.keras.constraints import maxnorm;
    from tensorflow.keras.layers.normalization import BatchNormalization
    from tensorflow.keras.regularizers import l1, l2    
    from tensorflow.keras import backend as K
    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    import collections
    model_inputs = ["data/genome_data_dir"]
    shapes = {'data/genome_data_dir': [1000, 4]}
    keras_inputs = collections.OrderedDict([(name, Input(shape=shapes[name], name=name)) for name in model_inputs])
    inputs = keras_inputs
    num_tasks = ntasks
    seq_preds = inputs["data/genome_data_dir"]
    num_filters = (48, 64, 100, 150, 300, 200, 200, 200, 200)
    conv_width = (3, 3, 3, 7, 7, 7, 3, 3, 7)
    batch_norm = True
    pool_width=(3, 4, 4)
    pool_stride=(3, 4, 4)
    fc_layer_sizes=(1000, 1000)
    dropout=(0.3, 0.3)
    final_dropout=0.0,
    trainable=1
    final_layer_name='tuned_i_score'
    j = 0
    for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
        seq_preds = Conv1D(nb_filter, nb_col, kernel_initializer='he_normal', trainable = bool(trainable))(seq_preds)
        if batch_norm:
            seq_preds = BatchNormalization(trainable = bool(trainable))(seq_preds)
        seq_preds = Activation('relu', trainable = bool(trainable))(seq_preds)

        if(i == 4 or i == 7 or i == 8):
            seq_preds = MaxPooling1D(pool_width[j], pool_stride[j], trainable = bool(trainable))(seq_preds)
            j = j+1

    seq_preds = Flatten()(seq_preds)

    # fully connect, drop before fc layers
    for drop_rate, fc_layer_size in zip(dropout, fc_layer_sizes):
        seq_preds = Dense(fc_layer_size)(seq_preds)
        if batch_norm:
            seq_preds = BatchNormalization()(seq_preds)
        seq_preds = Activation('relu')(seq_preds)
    seq_preds = Dense(num_tasks, name=final_layer_name)(seq_preds)
    seq_preds = Activation('sigmoid')(seq_preds)
    random_weight_model = Model(inputs=list(keras_inputs.values()), outputs=seq_preds)
    model = random_weight_model

    if (init_weights!=None):
        #load the weight initializations
        model.load_weights(init_weights, by_name=True)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    if w0!=None:
        loss=get_weighted_binary_crossentropy(w0_weights=w0,w1_weights=w1)
    else:
        loss=get_ambig_binary_crossentropy() 
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=[recall, specificity, fpr, fnr, precision, f1])
    return model
