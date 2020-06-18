import numpy as np
import os, librosa
from read_csv import read_csv
from audio_to_mel_magnitude_spectrogram import audio_to_mel_magnitude_spectrogram
from tensorflow.keras.utils import to_categorical

def make_features(dataset_dir = 'datasets' + os.sep, feature_dir = 'features' + os.sep):
    '''
    Helper function to make non-augmented features
    from downloaded DCASE dataset.
    
    ========
     Inputs
    ========
    
    dataset_dir : str
        The directory that the DCASE dataset was
        saved to.
    
    feature_dir : str
        The directory to save the output features
        to.
    
    =========
     Outputs
    =========
    
    Nothing. However the function generates .npy
    files from the downloaded dataset and saves
    them into feature_dir
    
    ==============
     Dependencies
    ==============
    
    numpy (as np), os, librosa, to_categorical
    (from tensorflow.keras.utils)
    
    read_csv, audio_to_mel_magnitude_spectrogram
    (our own functions)
    '''
    ## INITIALISE SOME VARIABLES
    train_feature_path = feature_dir + os.sep + 'raw_train_features.npy'
    train_label_path = feature_dir + os.sep + 'raw_train_labels.npy'
    eval_feature_path = feature_dir + os.sep + 'raw_eval_features.npy'
    eval_label_path = feature_dir + os.sep + 'raw_eval_labels.npy'
    test_feature_path = feature_dir + os.sep + 'raw_test_features.npy'
    class_labels = ['indoor','outdoor','transportation']
    
    ## MAKE FEATURE DIRECTORY IF IT DOESN'T ALREADY EXIST
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    
    try:
        ## CHECK VALIDITY OF TRAINING FEATURES (IF THEY ALREADY EXIST)
        train_features = np.load(train_feature_path, allow_pickle = True)
        train_labels = to_categorical(np.load(train_label_path,allow_pickle=True))
        assert train_features.shape == (9185,48,467,1) and train_labels.shape == (9185,3)
        del train_features, train_labels
        print('Training features and labels already exist.')
    except:
        print('Training features and labels don\'t already exist or error in validating them. Making them now...')
        
        ## GET METADATA FILES FROM DOWNLOADED DATASET
        train_files = read_csv(dataset_dir + 'TAU-urban-acoustic-scenes-2020-3class-development' + os.sep + 'evaluation_setup' + os.sep + 'fold1_train.csv',
                       delimiter = '\t')
        train_files.pop(0) # 9185 files
        
        ## MAKE FEATURES FOR TRAINING SET
        train_features = np.empty(shape=(len(train_files),48,467,1))
        for i in range(len(train_files)):
            filepath = dataset_dir + os.sep + 'TAU-urban-acoustic-scenes-2020-3class-development' + os.sep + train_files[i][0]
            x, sr = librosa.load(filepath,mono = True,sr = 48000)
            s = audio_to_mel_magnitude_spectrogram(input_data = x, sr = 48000) # By changing the arguments to this function we can output spectrograms with different dimensions.
            train_features[i] = s
            if i%100 == 0:
                print('Now on training sample: ' + str(i) + '/' + str(len(train_files)), end = '\r')
   
        ## MAKE LABELS FOR TRAINING SET
        train_labels = np.empty(shape = (len(train_files),))
        for i in range(len(train_files)):
            train_labels[i] = class_labels.index(train_files[i][1])
        
        ## SAVE FEATURES + LABELS FOR TRAINING SET
        np.save(train_feature_path,train_features,allow_pickle=True)
        np.save(train_label_path,train_labels,allow_pickle=True)
        del train_features, train_labels # To save memory
        
    try:
        ## CHECK VALIDITY OF VALIDATION FEATURES (IF THEY ALREADY EXIST)
        eval_features = np.load(eval_feature_path, allow_pickle = True)
        eval_labels = to_categorical(np.load(eval_label_path,allow_pickle=True))
        assert eval_features.shape == (4185,48,467,1) and eval_labels.shape == (4185,3)
        del eval_features, eval_labels
        print('Validation features and labels already exist.')
    except:
        print('Validation features and labels don\'t already exist or error in validating them. Making them now...')
        
        ## GET METADATA FILES FROM DOWNLOADED DATASET
        eval_files = read_csv(dataset_dir + 'TAU-urban-acoustic-scenes-2020-3class-development' + os.sep + 'evaluation_setup' + os.sep + 'fold1_evaluate.csv',
                              delimiter = '\t')
        eval_files.pop(0) # 4185 files
        
        ## MAKE FEATURES FOR VALIDATION SET
        eval_features = np.empty(shape=(len(eval_files),48,467,1))
        for i in range(len(eval_files)):
            filepath = dataset_dir + os.sep + 'TAU-urban-acoustic-scenes-2020-3class-development' + os.sep + eval_files[i][0]
            x, sr = librosa.load(filepath,mono = True,sr = 48000)
            s = audio_to_mel_magnitude_spectrogram(input_data = x, sr = 48000)
            eval_features[i] = s
            if i%100 == 0:
                print('Now on evaluation sample: ' + str(i) + '/' + str(len(eval_files)), end = '\r')
   
        ## MAKE LABELS FOR VALIDATION SET
        eval_labels = np.empty(shape = (len(eval_files),))
        for i in range(len(eval_files)):
            eval_labels[i] = class_labels.index(eval_files[i][1])
        
        ## SAVE FEATURES + LABELS FOR VALIDATION SET
        np.save(eval_feature_path,eval_features,allow_pickle=True)
        np.save(eval_label_path,eval_labels,allow_pickle=True)
        del eval_features, eval_labels # To save memory

    try:
        ## CHECK VALIDITY OF TEST FEATURES (IF THEY ALREADY EXIST)
        test_features = np.load(test_feature_path, allow_pickle = True)
        assert test_features.shape == (8640,48,467,1)
        del test_features
        print('Test features already exist.')
        ## NOTE: NO TEST LABELS PROVIDED FOR DCASE 2020 CHALLENGE.
    except:
        print('Test features don\'t already exist or error in validating them. Making them now...')
        test_files = read_csv(dataset_dir + 'TAU-urban-acoustic-scenes-2020-3class-evaluation' + os.sep + 'evaluation_setup' + os.sep + 'fold1_test.csv',
                               delimiter = '\t')
        test_files.pop(0) # 8640 files
        
        ## MAKE FEATURES FOR TEST SET
        test_features = np.empty(shape=(len(test_files),48,467,1))
        for i in range(len(test_files)):
            filepath = dataset_dir + os.sep + 'TAU-urban-acoustic-scenes-2020-3class-evaluation' + os.sep + test_files[i][0]
            x, sr = librosa.load(filepath,mono = True,sr = 48000)
            s = audio_to_mel_magnitude_spectrogram(input_data = x, sr = 48000)
            test_features[i] = s
            if i%100 == 0:
                print('Now on test sample: ' + str(i) + '/' + str(len(test_files)), end = '\r')
        
        ## SAVE FEATURES FOR TEST SET
        np.save(test_feature_path,test_features,allow_pickle=True)