import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, BatchNormalization, concatenate, AveragePooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow_model_optimization.sparsity import keras as sparsity # Pruning API for TF2.

__version__ = '20200615_dcase2020'

def inception_prunable(input_shape = (48,467,1), n_classes = 3, summary = False, **pruning_params):
    '''
    Defines the (prunable) InceptionNet-based model
    (with some variable parameters) for a single-
    label classification task.
    
    ========
     Inputs
    ========

    input_shape : tuple
        The input shape for the model (inputs are 
        usually spectrograms represented as a tensor
        of 3 dimensions)
        
    n_classes : int
        The number of classes in the classification
        task.

    summary : bool
        If True, calls model.summary to print out a
        summary of the model. If False, prints
        nothing.
 
    **pruning_params : dict
        Additional keyword arguments for the sparsity
        pruning API. For example, we could have:
        pruning_params = {
          'pruning_schedule':
          sparsity.PolynomialDecay(
          initial_sparsity=0.30,
          final_sparsity=0.90,begin_step=3000,
          end_step=14400,frequency=300)
        }
    
    =========
     Outputs
    =========
    
    model : tensorflow.python.keras.engine.
            training.Model
        The Keras model with the specified 
        InceptionNet-based architecture.
    
    ==============
     Dependencies
    ==============
    
    tensorflow.keras (as keras),
    tensorflow_model_optimization.sparsity.keras
    (as sparsity)
    
    Written by Kenneth Ooi
    '''
    X_input = Input(shape=input_shape)
    X = sparsity.prune_low_magnitude(Conv2D(16, (7, 7), padding='same', strides=(2, 2), activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_input)
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    X = BatchNormalization()(X)

    X = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X)
    X = sparsity.prune_low_magnitude(Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X)
    X = BatchNormalization()(X)

    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    ## 1st block
    tower_1 = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X)
    tower_1 = sparsity.prune_low_magnitude(Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_1)

    tower_2 = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X)
    tower_2 = sparsity.prune_low_magnitude(Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
    tower_3 = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_3)

    X_concat_1 = concatenate([tower_1, tower_2, tower_3], axis=3)

    ## 2nd block
    tower_4 = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_1)
    tower_4 = sparsity.prune_low_magnitude(Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_4)

    tower_5 = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_1)
    tower_5 = sparsity.prune_low_magnitude(Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_5)

    tower_6 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X_concat_1)
    tower_6 = sparsity.prune_low_magnitude(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_6)

    X_concat_2 = concatenate([tower_4, tower_5, tower_6], axis=3)

    ## 3rd block
    tower_7 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_2)
    tower_7 = sparsity.prune_low_magnitude(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_7)

    tower_8 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_2)
    tower_8 = sparsity.prune_low_magnitude(Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_8)

    tower_9 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X_concat_2)
    tower_9 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_9)

    X_concat_3 = concatenate([tower_7, tower_8, tower_9], axis=3)
    X_concat_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X_concat_3)

    ## 4th block
    tower_10 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_3)
    tower_10 = sparsity.prune_low_magnitude(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_10)

    tower_11 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_3)
    tower_11 = sparsity.prune_low_magnitude(Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_11)

    tower_12 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X_concat_3)
    tower_12 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_12)

    X_concat_4 = concatenate([tower_10, tower_11, tower_12], axis=3)

    ## 5th block
    tower_13 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_4)
    tower_13 = sparsity.prune_low_magnitude(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_13)

    tower_14 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X_concat_4)
    tower_14 = sparsity.prune_low_magnitude(Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_14)

    tower_15 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X_concat_4)
    tower_15 = sparsity.prune_low_magnitude(Conv2D(32, (1, 1), padding='same', activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(tower_15)

    X_concat_5 = concatenate([tower_13, tower_14, tower_15], axis=3)
    X_concat_5 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X_concat_5)

    X = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(X_concat_5)
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = sparsity.prune_low_magnitude(Dense(20, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)), **pruning_params)(X)
    X = Dropout(0.5)(X)
    X = sparsity.prune_low_magnitude(Dense(n_classes, activation='softmax'), **pruning_params)(X)
    model = Model(inputs=X_input, outputs=X)
    
    if summary:
        model.summary() # Print model summary if desired
    
    return model
