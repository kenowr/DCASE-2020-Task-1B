import numpy as np
np.random.seed(2020)

import os, glob
import tensorflow
import tensorflow.keras as keras
tensorflow.random.set_seed(2020)

from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.sparsity import keras as sparsity # Pruning API
from dcase2020task1b_functions import *
from model_size_calculation import get_keras_model_size

## INITIAL PREPARATION
download_dcase_dataset()
make_features()

## LOAD FEATURES
feature_dir = 'features' + os.sep
train_features = np.load(feature_dir + 'raw_train_features.npy',allow_pickle=True)
train_labels = to_categorical(np.load(feature_dir + 'raw_train_labels.npy',allow_pickle=True))
eval_features = np.load(feature_dir + 'raw_eval_features.npy',allow_pickle=True)
eval_labels = to_categorical(np.load(feature_dir + 'raw_eval_labels.npy',allow_pickle=True))

## DEFINE TRAINING PARAMETERS
batch_size = 128
n_batches = len(train_features)//batch_size + 1*(len(train_features)%batch_size!=0)
model_path = 'models' + os.sep + 'model3' + os.sep

## DEFINE PRUNING PARAMETERS
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.1,
                                                   final_sparsity=0.8,
                                                   begin_step=n_batches*100, # Begin pruning at epoch 100
                                                   end_step=n_batches*300, # End pruning at epoch 300 (but continue to train for 100 more epochs)
                                                   frequency=n_batches*10)
}

## GENERATE INCEPTION MODEL (SUBSYSTEM 1)
model = inception_prunable(**pruning_params)
print('Generated Model 3, Subsystem 1. Training it now...')

## TRAIN INCEPTION MODEL
model_hist = trainp_model(model,train_features,train_labels,eval_features,eval_labels,
                          string = 'MODEL3_SUBSYSTEM1_', model_path = model_path, batch_size = batch_size)

for subsystem in range(2,7):
    ## GENERATE VGG MODEL (SUBSYSTEM 2-6)
    model = vgg_prunable(**pruning_params)
    print('Generated Model 3, Subsystem {}. Training it now...'.format(subsystem))

    ## TRAIN VGG MODEL
    model_hist = trainp_model(model,train_features,train_labels,eval_features,eval_labels,
                              string = 'MODEL3_SUBSYSTEM{}_'.format(subsystem), model_path = model_path, batch_size = batch_size)

## FIND BEST SUBSYSTEMS
best_filepaths = []
for subsystem in range(1,7):
    model_list = glob.glob('models' + os.sep + 'model3' + os.sep + 'weights' + os.sep + 'MODEL3_SUBSYSTEM{}_epoch-[3,4]*'.format(subsystem)) # Get models from epoch 300-400 only because they are pruned.
    _, best_filepath, _, _ = get_best_model(eval_features, eval_labels, model_list)
    print('Best filepath for subsystem {} is : '.format(subsystem) + best_filepath)
    best_filepaths.append(best_filepath)
    
## MAKE PREDICTIONS ON TEST SET
test_features = np.load(feature_dir + 'raw_test_features.npy',allow_pickle=True)
submission_dir = 'output_submission' + os.sep
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)
_ = make_predictions(test_features, best_filepaths, output_filepath = submission_dir + 'Ooi_NTU_task1b_3.output.csv', delimiter = '\t', newline = '\n')
print('Test set predictions saved in ' + submission_dir)

## GET METRICS
n_nz_params = 0
model_size = 0
for best_filepath in best_filepaths:
    with sparsity.prune_scope(): # Need to use this to prevent loading errors
        keras_model = keras.models.load_model(best_filepath, compile = False) # Don't compile model to save time because we're not training it here.
    keras_model = sparsity.strip_pruning(keras_model)
    param_dict = get_keras_model_size(keras_model,verbose=False)
    n_nz_params += param_dict['parameters']['non_zero']['count']
    model_size += param_dict['parameters']['non_zero']['bytes']/1024

output_meta = make_predictions(eval_features, best_filepaths, save = False)
best_micro_acc, best_macro_acc = accs(np.array([row[-3:] for row in output_meta[1:]]), eval_labels)

## PRINT FINAL METRICS
print()
print('============================')
print(' FINAL ANALYSIS FOR MODEL 3 ')
print('============================')
print()
print('Model files used: ')
for best_filepath in best_filepaths:
    print('\t' + best_filepath)
print('Micro-averaged accuracy: {:.4f}'.format(best_micro_acc))
print('Macro-averaged accuracy: {:.4f}'.format(best_macro_acc))
print('# non-zero parameters: {:d}'.format(n_nz_params))
print('Size/KB: {:.1f}'.format(model_size))
