import numpy as np ;
from kerasAC.metrics import *
from kerasAC.custom_losses import *

def get_model(args):
    from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1    
    custom_objects={"recall":recall,
                    "sensitivity":recall,
                    "specificity":specificity,
                    "fpr":fpr,
                    "fnr":fnr,
                    "precision":precision,
                    "f1":f1,
                    "ambig_binary_crossentropy":ambig_binary_crossentropy,
                    "ambig_mean_absolute_error":ambig_mean_absolute_error,
                    "ambig_mean_squared_error":ambig_mean_squared_error}
    if args.from_checkpoint_arch is not None:
        from keras.models import model_from_json
        #load the model architecture from json
        json_string=open(args.from_checkpoint_arch,'r').read()
        model=model_from_json(json_string,custom_objects=custom_objects)
    if args.init_weights!=None: 
        #load from the hdf5
        from keras.models import load_model
        model=load_model(args.model_hdf5,custom_objects=custom_objects)
    return model

def get_model_weights(args,model):
    assert args.from_checkpoint_weights is not None
    model.load_weights(args.from_checkpoint_weights,by_name=True)
    return model


def getModelGivenModelOptionsAndWeightInits(args):
    #get the arguments
    seed=args.seed
    w0=args.w0
    w1=args.w1
    init_weights=args.init_weights
    ntasks=args.num_tasks
    
    np.random.seed(seed)
    import keras;
    from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.optimizers import Adadelta, SGD, RMSprop;
    import keras.losses;
    from keras.constraints import maxnorm;
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1, l2
    from keras import backend as K
    from keras.layers import Input, Add
    from keras.models import Model

    K.set_image_data_format('channels_last')
    print(K.image_data_format())
    if args.init_weights is True:
        #load model with custom objects 
        model=get_model(args)
    elif args.from_checkpoint_arch is True :
        model=get_model(args)
    else: 
        seq = Input(shape=(1000,4),name='input')
        x = Conv1D(filters=300,kernel_size=19,input_shape=(1000,4),name='conv1')(seq)
        x = BatchNormalization(axis=-1,name='bn1')(x)
        x = Activation('relu',name='relu1')(x)
        x = MaxPooling1D(pool_size=3,name='maxpool1')(x)
        
        x = Conv1D(filters=200,kernel_size=11,name='conv2')(x)
        x = BatchNormalization(axis=-1,name='bn2')(x)
        x = Activation('relu',name='relu2')(x)
        x = MaxPooling1D(pool_size=4,name='maxpool2')(x)
        
        x = Conv1D(filters=200,kernel_size=7,name='conv3')(x)
        x = BatchNormalization(axis=-1,name='bn3')(x)
        x = Activation('relu',name='relu3')(x)
        x = MaxPooling1D(pool_size=4,name='maxpool3')(x)
        
        x = Flatten(name='flatten1')(x)
        x = Dense(1000,name='dense1')(x)
        x = BatchNormalization(axis=-1,name='bn4')(x)
        x = Activation('relu',name='relu4')(x)
        x = Dropout(0.3,name='dropout1')(x)

        x = Dense(1000,name='dense2')(x)
        x = BatchNormalization(axis=-1,name='bn5')(x)
        x = Activation('relu',name='relu5')(x)
        x = Dropout(0.3,name='dropout2')(x)
        outputs= Dense(ntasks,name='final_dense'+str(ntasks))(x)
        model = Model(inputs = [seq], outputs = outputs)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    loss=ambig_mean_squared_error
    model.compile(optimizer=adam,loss=loss)
    
    if args.from_checkpoint_weights is True:
        model=get_model_weights(args,model=model)
    return model,adam,loss

