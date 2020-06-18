import numpy as np
import os, librosa
from read_csv import read_csv
from audio_to_mel_magnitude_spectrogram import audio_to_mel_magnitude_spectrogram
from tensorflow.keras.utils import to_categorical

__version__ = '20200615_dcase2020'

def augment_features(dataset_dir = 'datasets' + os.sep, feature_dir = 'features' + os.sep,
                     samples_per_class = 4000, seed_no = 2020):
    '''
    Helper function to make augmented features
    from training set of downloaded DCASE raw
    audio data.
    
    ========
     Inputs
    ========
    
    dataset_dir : str
        The directory that the DCASE dataset was
        saved to.
    
    feature_dir : str
        The directory to save the output features
        to.
        
    samples_per_class : int
        The number of augmented training samples
        to generate for each class in the dataset.
    
    seed_no : int
        The seed to pass to np.random.seed for
        generating random blocks to mix.
    
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
    augmented_feature_path = feature_dir + os.sep + 'augmented_train_features.npy'
    augmented_label_path = feature_dir + os.sep + 'augmented_train_labels.npy'
    class_labels = ['indoor','outdoor','transportation']
    np.random.seed(seed_no)
    track_length = 10 # in seconds
    
    ## MAKE FEATURE DIRECTORY IF IT DOESN'T ALREADY EXIST
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    
    try:
        ## CHECK VALIDITY OF AUGMENTED FEATURES (IF THEY ALREADY EXIST)
        augmented_features = np.load(augmented_feature_path, allow_pickle = True)
        augmented_labels = to_categorical(np.load(augmented_label_path,allow_pickle=True))
        assert (augmented_features.shape[1],augmented_features.shape[2],augmented_features.shape[3]) == (48,467,1) and augmented_labels.shape[1] == 3
        assert augmented_features.shape[0] == augmented_labels.shape[0]
        if augmented_features.shape[0] != 12000:
            print('Warning: Number of augmented samples does not match the default setting.')
        del augmented_features, augmented_labels
        print('Augmented features and labels already exist.')
    except:
        ## READ TRAINING METADATA
        train_files = read_csv(dataset_dir + os.sep + 'TAU-urban-acoustic-scenes-2020-3class-development' + os.sep + 'evaluation_setup' + os.sep + 'fold1_train.csv',
                               delimiter = '\t')
        train_files.pop(0)

        ## ADD CITY NAMES TO METADATA
        for i in train_files:
            i.append(i[0].split('-')[1])

        ## GET LIST OF CITY NAMES
        cities = []
        for i in train_files:
            if i[2] not in cities:
                cities.append(i[2]) # Should be ['lisbon','lyon','prague','barcelona','helsinki','london','paris', 'stockholm','vienna']

        ## SORT TRACKS BY CITY 
        sorted_list = []
        for i in range(len(cities)): # Prepare empty list of lists
            elem = []
            for i in range(len(class_labels)):
                elem.append([])
            sorted_list.append(elem)

        for i in train_files: # Add in entries
            try:
                sorted_list[cities.index(i[2])][class_labels.index(i[1])].append(i)
            except:
                print('Not found in list: ' + str(i))

        ## MAKE AUGMENTED FEATURES AND LABELS
        city_idx = 0 # Counts current city (in circular manner)
        counter = 0 # Counts output spectrograms
        augmented_features = np.empty(shape=(samples_per_class*len(class_labels),48,467,1))
        augmented_labels = np.empty(shape=(samples_per_class*len(class_labels),))

        for class_idx in range(len(class_labels)):
            for i in range(samples_per_class):
                
                ## GET LIST OF FILES FROM WHICH TO EXTRACT 1-SECOND TRACKS
                subfiles = []
                for j in range(track_length):
                    file_idx = np.random.randint(len(sorted_list[city_idx][class_idx]))
                    filepath = dataset_dir + os.sep + 'TAU-urban-acoustic-scenes-2020-3class-development' + os.sep + sorted_list[city_idx][class_idx][file_idx][0]
                    filepath = filepath.replace('/',os.sep) # Replace Linux-style directory separator with native directory separator
                    subfiles.append(filepath)

                    city_idx += 1
                    if city_idx >= len(cities): # 9
                        city_idx = 0
                subfiles = list(np.random.permutation(subfiles))

                ## EXTRACT 1S TRACKS AND MAKE 10S FILE
                outfile = np.empty(shape=48000*10)
                for sec, filepath in enumerate(subfiles):
                    x, sr = librosa.load(filepath,mono = True,sr = 48000)
                    start_index = np.random.randint(48000*(10-1)+1)
                    outfile[sec*48000:(sec+1)*48000] = x[start_index:start_index+48000]
                s = audio_to_mel_magnitude_spectrogram(input_data = outfile, sr = 48000)
                augmented_features[counter] = s
                augmented_labels[counter] = class_idx
                counter += 1

                if counter%100 == 0:
                    print('Now making augmented sample: ' + str(counter) + '/' + str(augmented_features.shape[0]), end = '\r')

        ## SAVE AUGMENTED FEATURES AND LABELS
        np.save(feature_dir + os.sep + 'augmented_train_features.npy',augmented_features,allow_pickle=True)
        np.save(feature_dir + os.sep + 'augmented_train_labels.npy',augmented_labels,allow_pickle=True)