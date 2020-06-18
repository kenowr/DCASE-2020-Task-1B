import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import tensorflow.keras as keras
from datetime import datetime
from tensorflow_model_optimization.sparsity import keras as sparsity # Import the pruning API

__version__ = '20200615_dcase2020'

def trainp_model(model, train_features, train_labels, eval_features, eval_labels,
                 string = '', prepend_date = False, model_path = 'models' + os.sep + 'pruned' + os.sep,
                 patience = 400, epochs = 400, initial_epoch = 0, batch_size = 128, learning_rate = 0.0001,
                 monitor = 'val_accuracy', save_best_only = False, prune_model = True,
                 verbose_checkpoint = 1, verbose_fit = 1):
    '''
    Train and prunes a Keras model with given
    parameters.

    ========
     Inputs
    ========
    
    model : tensorflow.python.keras.engine.training.
            Model or tensorflow.python.keras.engine.
            sequential.Sequential
        A Keras model to be trained
        
    train_featres : numpy.ndarray
        Training features acting as input to the
        Keras model.fit command.

    train_labels : numpy.ndarray
        Training labels (one-hot categorical) acting
        as ground truth to the Keras model.fit
        command.
        
    eval_features : numpy.ndarray
        Evaluation (validation) features for the
        Keras model.fit command.
        
    eval_labels : numpy.ndarray
        Evaluation (validation) labels (one-hot
        categorical) for the Keras model.fit command.

    string : str
        An optional string to prepend to each of the
        output files for identification
    
    prepend_date : bool
        If true, prepends date to output files.

    model_path : str
        Folder path to save figures, model weights,
        and model histories to. Will be made if it 
        doesn't already exist

    patience : int
        Number of epochs to wait if no improvement
        occurs before stopping (same as the argument
        for keras.callbacks.EarlyStopping)
        
    epochs : int
        Number of epochs to train the model for (same
        as the argument for model.fit)

    initial_epoch : int
        Starting epoch number (same as the argument
        for model.fit)
        
    batch_size : int
        Batch size for training. Same as the argument
        for model.fit
        
    learning_rate : float
        The learning rate for the adam optimiser used
        in model.compile (keras.optimizers.Adam)
        
    monitor : str or callable
        The metric to monitor for patience tracking
        (same as the arguments in keras.callbacks.
        ModelCheckpoint and keras.callbacks.
        EarlyStopping)
        
    save_best_only : bool
        Same as the argument in keras.callbacks.
        ModelCheckpoint
   
    prune_model : bool
        Whether to prune the model accepted as input
        to this function. Note that the pruning
        parameters should already be present in the 
        generation of the model object if using the
        Keras tfmot API.
        
    verbose_checkpoint : bool
        The value of verbose in
        keras.callbacks.ModelCheckpoint
    
    verbose_fit : bool
        The value of verbose in model.fit.
    
    =========
     Outputs
    =========
    
    model_hist : 
        A Keras history object containing the
        trained model and relevant metadata.
    
    ==============
     Dependencies
    ==============
    
    os, pickle, matplotlib.pyplot (as plt), numpy
    (as np), tensorflow.keras (as keras), datetime,
    tensorflow_model_optimization.sparsity.keras
    (as sparsity)
    '''
    
    ## COMPILE MODEL
    model.compile(loss = 'categorical_crossentropy', # Categorical crossentropy is a standard function for single-label classification. 
                  optimizer = keras.optimizers.Adam(lr = learning_rate), # The best optimiser to use might depend on your data, but in general the Adam optimiser performs pretty well.
                  metrics = ['accuracy']) # Is NOT optimised over, but is something that is printed when running model.fit and saved in model_history object (explained later).

    ## PREPARE OUTPUT DIRECTORIES
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(model_path + os.sep + 'weights'):
        os.makedirs(model_path + os.sep + 'weights')
    if not os.path.exists(model_path + os.sep + 'figures'):
        os.makedirs(model_path + os.sep + 'figures')
    if not os.path.exists(model_path + os.sep + 'histories'):
        os.makedirs(model_path + os.sep + 'histories')

    ## DEFINE CALLBACKS
    if prepend_date:
        filepath = model_path + os.sep + 'weights' + os.sep + str(datetime.now()).split(' ')[0] + '_' + string + 'epoch-{epoch:03d}-loss-{val_loss:.4f}-acc-{val_accuracy:.4f}.h5' 
    else:
        filepath = model_path + os.sep + 'weights' + os.sep + string + 'epoch-{epoch:03d}-loss-{val_loss:.4f}-acc-{val_accuracy:.4f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                 monitor = monitor,
                                                 verbose = verbose_checkpoint,
                                                 save_best_only = save_best_only) 
    early = keras.callbacks.EarlyStopping(monitor = monitor, mode = "min", patience = patience) 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir = model_path + os.sep + 'weights' + os.sep,
                                                       histogram_freq = 0, write_images = False) # Only if using Tensorboard
    
    if prune_model:
        sparsity_callback = sparsity.UpdatePruningStep()
        callbacks_list = [checkpoint, early, tensorboard_callback, sparsity_callback]
    else:
        callbacks_list = [checkpoint, early, tensorboard_callback]
    
    ## FIT MODEL
    model_hist = model.fit(train_features,
                           train_labels,
                           batch_size = batch_size,
                           epochs = epochs,
                           initial_epoch = initial_epoch, # The label of the first epoch upon calling this function is one greater than this.
                                                          # Call a number k > 0 to label the epochs starting from k+1 (useful for resuming training).
                           verbose = verbose_fit,
                           validation_data = (eval_features, eval_labels),
                           callbacks = callbacks_list,
                           shuffle = True)
    
    ## MAKE ACCURACY PLOT
    max_train_acc = np.max(model_hist.history['accuracy'])
    max_train_epoch = np.argmax(model_hist.history['accuracy'])
    max_valid_acc = np.max(model_hist.history['val_accuracy'])
    max_valid_epoch = np.argmax(model_hist.history['val_accuracy'])

    plt.figure(figsize=(15,10))
    plt.title('Classification accuracy (micro)', fontsize = 24)

    p1 = plt.plot(np.array(model_hist.epoch), model_hist.history['accuracy'], label = 'Training ({:.4f})'.format(max_train_acc), linewidth = 2)
    p2 = plt.plot(np.array(model_hist.epoch), model_hist.history['val_accuracy'], label = 'Validation ({:.4f})'.format(max_valid_acc), linewidth = 2)

    xl, xr = plt.xlim()
    yl, yr = plt.ylim()

    plt.hlines(max_train_acc, xl, xr, linestyles = 'dotted', colors = p1[0].get_color(), linewidth = 2)
    plt.hlines(max_valid_acc, xl, xr, linestyles = 'dotted', colors = p2[0].get_color(), linewidth = 2)
    plt.vlines(max_train_epoch, yl, yr, linestyles = 'dotted', colors = p1[0].get_color(), linewidth = 2)
    plt.vlines(max_valid_epoch, yl, yr, linestyles = 'dotted', colors = p2[0].get_color(), linewidth = 2)

    plt.xlim(xl,xr) # Reset window to original limits
    plt.ylim(yl,yr)

    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Value', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 18)
    plt.grid(which = 'both')
    plt.savefig(model_path + os.sep + 'figures' + os.sep +string+'accuracy.png', transparent = True, bbox_inches='tight')
    
    ## MAKE LOSS PLOT
    min_train_loss = np.min(model_hist.history['loss'])
    min_train_epoch = np.argmin(model_hist.history['loss'])
    min_valid_loss = np.min(model_hist.history['val_loss'])
    min_valid_epoch = np.argmin(model_hist.history['val_loss'])
    
    plt.figure(figsize=(15,10))
    plt.title('Classification loss', fontsize = 24)

    p1 = plt.plot(np.array(model_hist.epoch), model_hist.history['loss'], label = 'Training ({:.4f})'.format(min_train_loss), linewidth = 2)
    p2 = plt.plot(np.array(model_hist.epoch), model_hist.history['val_loss'], label = 'Validation ({:.4f})'.format(min_valid_loss), linewidth = 2)

    xl, xr = plt.xlim()
    yl, yr = plt.ylim()

    plt.hlines(min_train_loss, xl, xr, linestyles = 'dotted', colors = p1[0].get_color(), linewidth = 2)
    plt.hlines(min_valid_loss, xl, xr, linestyles = 'dotted', colors = p2[0].get_color(), linewidth = 2)
    plt.vlines(min_train_epoch, yl, yr, linestyles = 'dotted', colors = p1[0].get_color(), linewidth = 2)
    plt.vlines(min_valid_epoch, yl, yr, linestyles = 'dotted', colors = p2[0].get_color(), linewidth = 2)

    plt.xlim(xl,xr) # Reset window to original limits
    plt.ylim(yl,yr)

    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Value', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 18)
    plt.grid(which = 'both')
    plt.savefig(model_path + os.sep + 'figures' + os.sep +string+'loss.png', transparent = True, bbox_inches='tight')
    
    ## SAVE HISTORY
    with open(model_path + os.sep + 'histories' + os.sep + 'History_' + string + '{:.4f}-epoch-{:03d}.pickle'.format(
        np.max(model_hist.history['val_accuracy']),np.argmax(model_hist.history['val_accuracy'])+1
        ),'wb') as pickle_out:
        pickle.dump((model_hist.history, model_hist.epoch), pickle_out)
    
    return model_hist