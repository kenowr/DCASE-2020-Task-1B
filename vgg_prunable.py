import tensorflow.keras as keras # For TensorFlow versions < 2.0.0, replace with the line 'import keras' instead.
from tensorflow_model_optimization.sparsity import keras as sparsity # Pruning API for TF2.

__version__ = '20200615_dcase2020'

def vgg_prunable(input_shape = (48,467,1), n_classes = 3, filter_seq = (12,16,20,24,28),
                 filter_size = (3,3), pool_size = (2,2), dropout_rate = 0.5, n_dense = 20, summary = False,
                 **pruning_params):
    '''
    Defines the (prunable) VGGNet-based model
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
        
    filter_seq : tuple of int
        A tuple of 5 elements representing the the
        number of convolutional filters for each of 
        the 5 VGG blocks in thbe model.
 
    filter_size : tuple of int
        A tuple of 2 elements representing the
        dimensions (height, width) of the filters used
        in all but the last convolutional layer.
        
    pool_size : tuple of int
        A tuple of 2 elements representing the
        dimensions (height, width) of the max pooling
        layers.

    dropout_rate : float
        A number between 0 and 1 specifying the rate
        of dropout for the dropout layers.
        
    n_dense : int
        An integer specifying the number of hidden
        layer neurons in the penultimate dense layer.

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
            sequential.Sequential
        The Keras model with the specified VGGNet-
        based architecture.
    
    ==============
     Dependencies
    ==============
    
    tensorflow.keras (as keras),
    tensorflow_model_optimization.sparsity.keras
    (as sparsity)
    
    Written by Kenneth Ooi
    '''
    model = keras.models.Sequential([
        ## 1st VGG block
        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[0], filter_size, padding='same', input_shape = input_shape, kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[0], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        
        ## 2nd VGG block
        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[1], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[1], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        
        ## 3rd VGG block
        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[2], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[2], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[2], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        
        ## 4th VGG block
        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[3], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[3], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),

        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[3], filter_size, padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=pool_size),
        
        ## 5th VGG block
        sparsity.prune_low_magnitude(keras.layers.Conv2D(filter_seq[4], (1,1), padding='same', kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        
        ## Dense and output layers
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout_rate),
        sparsity.prune_low_magnitude(keras.layers.Dense(n_dense, kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate),
        sparsity.prune_low_magnitude(keras.layers.Dense(n_classes, kernel_regularizer = keras.regularizers.l2(0.001)),
                                     **pruning_params), # This is the output layer.
        keras.layers.Activation('softmax'),
    ])

    if summary:
        model.summary() # Print model summary if desired
              
    return model