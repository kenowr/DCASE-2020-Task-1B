import csv, dcase_util, glob, IPython, librosa, librosa.display, os, pickle, wget, zipfile
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow.keras as keras # For TensorFlow versions < 2.0.0, replace with the line 'import keras' instead.
from datetime import datetime
from tensorflow_model_optimization.sparsity import keras as sparsity # Pruning API for TF2.
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, BatchNormalization, concatenate, AveragePooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import load_model, Model

__version__ = '20200615_dcase2020'

def accs(y_hat, y):
    '''
    Calculate micro-averaged accuracy and
    macro-averaged accuracy (average of
    class-wise accuracies) given predicted
    and ground-truth labels.
    
    ========
     Inputs
    ========
    
    y_hat : np.ndarray
        A (n_samples, n_classes) numpy array
        with the (i,j)-th element being the
        predicted probability that sample i
        belongs to class j.
    
    y : np.ndarray
        A (n_samples, n_classes) numpy array
        with the (i,j)-th element being 1 if
        the ground-truth label of sample i is
        class j and 0 otherwise.
    
    =========
     Outputs
    =========
    
    macro_avg_acc : float
        The macro-averaged accuracy of the
        predictions in y_hat.
              
    ==============
     Dependencies
    ==============
    
    numpy (as np)
    '''
    ## EXCEPTION HANDLING
    if y_hat.shape != y.shape:
        print('Error: y_hat and y have different dimensions.')
        return 0

    ## PREPARE PARAMS
    samples_by_class = np.sum(y,axis = 0) # Array of n_classes elements with # gt samples for each class
    n_classes = len(samples_by_class)
    y_hat = np.argmax(y_hat,axis=1)
    y = np.argmax(y,axis=1)

    ## CALCULATE CONFUSION MATRIX (BY SAMPLES)
    conf_mat = np.zeros(shape=(n_classes,n_classes))
    for (pred,actual) in zip(y_hat,y):
        conf_mat[pred][actual] += 1
        
    ## CALCULATE MICRO AND MACRO ACCURACY
    micro_avg_acc = np.trace(conf_mat)/np.sum(conf_mat) # Confusion matrix currently contains absolute numbers of samples as each element.
    conf_mat /= np.tile(samples_by_class,(n_classes,1)) # Normalise confusion matrix to probabilities.
    macro_avg_acc = np.trace(conf_mat)/n_classes
    
    return micro_avg_acc, macro_avg_acc

def audio_to_mel_magnitude_spectrogram(input_data = np.sin(np.linspace(0,440,44100)*np.pi), sr = None, n_fft = 2048, hop_length = 1024, center = False,
                                       n_mels = 48, fmin = 0, fmax = 24000, ref = 1.0,
                                       plot_spectrogram = False, titles = [], figsize = (20,4), fontsize = 20, vmin = None, vmax = None,
                                       **kwargs):
    '''
    ========
     Inputs
    ========
    
    input_data : str or np.ndarray
        If a string, it is the filepath of the input data (that
        will be read by sf.read).
        If it is an np.ndarray, it should be either a (n,)- or
        (c,n)-shaped array if it is single-channel or multi-
        channel respectively, where c is the number of channels in
        the signal and n is the number of samples in each channel.
        It will represent the signal in floating point numbers
        between -1 and 1. This function will convert a (n,)-
        shaped array to a (1,n)-shaped array while running.
        Default is a one-second 440Hz sine tone sampled at 44100Hz.
    
    sr : int
        The sampling rate of the signal specified by input_data.
        If it is not specified, then the native sampling rate of
        the file in input_data will be used (if input_data is a
        string) or a default sampling rate of 44100Hz will be used
        (if input_data is an np.ndarray).
    
    n_fft : int
        The number of samples in each time window for the
        calculation of the STFT of the input signal using
        librosa.core.stft.
        
    hop_length : int
        The number of overlapping samples in each time window for
        the calculation of the STFT of the input signal using
        librosa.core.stft.
    
    center : bool
        If True, centers the window at the current time index of
        the signal before performing FFT on it. If False, does not
        center the window, i.e. the current time index IS the 
        first index of the window. This is as per the parameter in
        librosa.core.stft.
        
    n_mels : int
        The number of mel bands used for calculation of the log-
        frequency power spectrom by librosa.feature.melspectrogram
    
    fmin : float
        The minimum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    
    fmax : float
        The maximum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    
    ref : float or callable
        The reference value or a function which returns the
        reference value for which dB computation is done in
        librosa.power_to_db.
    
    plot_spectrogram : bool
        If True, plots one spectrogram per channel in the signal
        specified in input_data. Otherwise, plots nothing.
    
    titles : list of str
        A list of c strings, where c is the number of channels in
        input_data, corresponding to the title of each mel
        spectrogram that is plotted. A default string will be
        assigned to each spectrogram if titles is an empty list.
        
    figsize : tuple of 2 int values
        If plot == True, this is the size of the spectrogram plot.
        If plot == False, this argument is ignored.
        
    fontsize : int
        The font size for the title of the plots, if plot == True.
        The font size of other elements in the plot will be 
        resized relative to this number. If plot == False, this
        argument is ignored.
    
    vmin : float or None
        The minimum value of the colour bar legend in the
        plt.clim() call. If None, the minimum of the input
        spectrogram values is used.
    
    vmax : float or None
        The maximum value of the colour bar legend in the
        plt.clim() call. If None, the maximum of the input
        spectrogram values is used.

    **kwargs : dict
        Extra keyword arguments to control the plot formatting
        with librosa.display.specshow (e.g. x_axis = 's',
        y_axis = 'mel', etc.)

    =========
     Outputs
    =========
    
    mel_spectrograms : np.ndarray
        An (n_mels, t, c)-shaped array containing the mel-
        spectrograms of each channel, where n_mels is the number
        of mel bands specified in the input argument to this
        function, t is the number of time bins in the STFT, and
        c is the number of channels in input_data.
        t depends on the input argument center as follows:
        If center is True, then t = np.ceil(n/hop_length).
        If center is False, then t = np.floor(n/hop_length)-1

    ==============
     Dependencies
    ==============
    
    librosa, librosa.display, matplotlib.pyplot (as plt),
    numpy (as np), soundfile (as sf)
    '''
    ## EXCEPTION HANDLING

    if type(input_data) == str: # If the input data entered is a string,...
        input_data, native_sr = sf.read(input_data) # ...then we read the filename specified in the string.
        input_data = np.transpose(input_data) # Transpose the input data to fit the (c,n)-shape desired.
        sr = native_sr if sr == None else sr
    elif type(input_data) == np.ndarray: # else we assume it is an np.ndarray
        sr = 44100 if sr == None else sr
    else:
        print('Invalid input data type! Input data must either be a string or a numpy.ndarray. Program terminating.')
        return


    # At this point, input_data should be either a (n,)- or (c,n)-shaped array.
    if len(input_data.shape) == 1: # If it's a (n,)-shaped array,...
        input_data = np.expand_dims(input_data,0) # ...then convert it into a (1,n)-shaped array.
    
    if len(titles) == 0:
        titles = ['Log-frequency power spectrogram (channel {:d})'.format(i) for i in range(input_data.shape[0])] # The default title for each spectrogram just iterates on the channel number.
    elif input_data.shape[0] != len(titles):
        print('Number of titles does not match number of channels in input data! Program terminating.')
        return
    
    if input_data.shape[0] > input_data.shape[1]:
        print('Warning: The input data appears to have more channels than samples. Perhaps you intended to input its transpose?')

    ## CALCULATE MEL SPECTROGRAM OF FIRST CHANNEL

    # Firstly, calculate the short-time Fourier transform (STFT) of the signal with librosa.core.stft.
    # We typecast input_data[0] as a Fortran-contiguous array because librosa.core.stft does vectorised operations on it,
    # and numpy array slices are typically not Fortran-contiguous.
    input_stft = librosa.core.stft(y = np.asfortranarray(input_data[0]), n_fft = n_fft, hop_length = hop_length, center = center)

    # Then, calculate the mel magnitude spectrogram of the STFT'd signal with librosa.feature.melspectrogram.
    power_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)

    # Convert the power spectrogram into into units of decibels with librosa.power_to_db.
    mel_spectrogram = librosa.power_to_db(power_spectrogram, ref = ref)

    # mel_spectrogram is an np.array. We typecast all elements to np.float32 to ensure all output data types match.
    mel_spectrogram = mel_spectrogram.astype(np.float32)


    ## PLOT MEL SPECTROGRAM OF FIRST CHANNEL

    if plot_spectrogram:
        plt.figure(figsize = figsize) # Sets size of plot.
        librosa.display.specshow(mel_spectrogram, sr = sr, hop_length = hop_length, fmin = fmin, fmax = fmax, **kwargs)
        plt.title(titles[0], fontsize = fontsize)         # Set title of figure.
        plt.xlabel('Time/s', fontsize = fontsize)         # Add in x-axis label to graph.
        plt.xticks(fontsize = 0.7*fontsize)               # Set font size for x-axis ticks (i.e. the numbers at each grid line).
        plt.ylabel('Frequency/Hz', fontsize = fontsize)   # Add in y-axis label to graph.
        plt.yticks(fontsize = 0.7*fontsize)               # Set font size for y-axis ticks (i.e. the numbers at each grid line).
        plt.colorbar(format='%+3.1f dB')                  # Adds in colour bar (legend) for values in spectrogram.
        plt.clim(vmin = vmin,vmax = vmax)                 # Defines the colour bar limits.
        plt.show()                                        # Display the actual plot on the IPython console.


    ## INITIALISE OUTPUT ARRAY (3-DIMENSIONAL) BASED ON FIRST CHANNEL MEL SPECTROGRAM SHAPE

    mel_spectrograms = np.zeros((mel_spectrogram.shape[0],mel_spectrogram.shape[1],input_data.shape[0]))
    mel_spectrograms[:,:,0] = mel_spectrogram # Put the first channel mel spectrogram in first.


    ## CALCULATE AND PLOT MEL SPECTROGRAM OF OTHER CHANNELS, IF ANY

    for i in range(1,input_data.shape[0]): # for i in 1:(number of channels),...
        input_stft = librosa.core.stft(y = np.asfortranarray(input_data[i]), n_fft = n_fft, hop_length = hop_length, center = center)
        power_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)
        mel_spectrogram = librosa.power_to_db(power_spectrogram, ref = ref)
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        mel_spectrograms[:,:,i] = mel_spectrogram # Put the calculated mel spectrogram as that of the ith channel.

        if plot_spectrogram:
            plt.figure(figsize = figsize) # Sets size of plot.
            librosa.display.specshow(mel_spectrogram, sr = sr, hop_length = hop_length, fmin = fmin, fmax = fmax, **kwargs)
            plt.title(titles[i], fontsize = fontsize)  # Set title of figure.
            plt.xlabel('Time/s', fontsize = fontsize)         # Add in x-axis label to graph.
            plt.xticks(fontsize = 0.7*fontsize)               # Set font size for x-axis ticks (i.e. the numbers at each grid line).
            plt.ylabel('Frequency/Hz', fontsize = fontsize)   # Add in y-axis label to graph.
            plt.yticks(fontsize = 0.7*fontsize)               # Set font size for y-axis ticks (i.e. the numbers at each grid line).
            plt.colorbar(format='%+3.1f dB')                  # Adds in colour bar (legend) for values in spectrogram.
            plt.clim(vmin = vmin,vmax = vmax)                 # Defines the colour bar limits.
            plt.show()                                        # Display the actual plot on the IPython console.

    return mel_spectrograms

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

def download_dcase_dataset(dataset_dir = 'datasets' + os.sep):
    '''
    Helper function to download DCASE dataset to
    a specified directory.
    
    ========
     Inputs
    ========
    
    dataset_dir : str
        The directory to save the DCASE dataset
        to.
    
    =========
     Outputs
    =========
    
    Nothing. However the function downloads and
    saves the DCASE dataset into dataset_dir.
    
    ==============
     Dependencies
    ==============
    
    zipfile, os, wget, glob,
    read_csv (our own function)
    '''
    ## CHECK IF DATASET ALREADY EXISTS
    dataset_filenames = read_csv('dataset_filenames.csv',list_of_str=True)
    complete_dataset_exists = True
    for dataset_filename in dataset_filenames:
        if not os.path.exists(dataset_filename):
            complete_dataset_exists = False

    if not complete_dataset_exists:
        print('One or more files in dataset does not exist. Downloading now...')

        ## PREPARE DATASET DIRECTORIES
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        ## GET DATASET URLS
        devt_urls = read_csv(filepath = 'devt_urls.csv', list_of_str = True)
        eval_urls = read_csv(filepath = 'eval_urls.csv', list_of_str = True)

        ## DOWNLOAD DEVELOPMENT SET
        while len(devt_urls) > 0:
            try:
                devt_url = devt_urls.pop()
                print('\nDownloading: ' + devt_url)
                wget.download(devt_url, out = dataset_dir)
            except:
                print('\nDownload failed. Trying again...')
                devt_urls.append(devt_url)

        ## UNZIP FILES AND DELETE ZIP ARCHIVES
        zipfiles = glob.glob(dataset_dir + os.sep + '*.zip')
        for filepath in zipfiles:
            with zipfile.ZipFile(filepath, 'r') as unzipper:
                unzipper.extractall(dataset_dir)
        for filepath in zipfiles:
            os.remove(filepath)

        ## DOWNLOAD EVALUATION SET
        while len(eval_urls) > 0:
            try:
                eval_url = eval_urls.pop()
                print('\nDownloading: ' + eval_url)
                wget.download(eval_url, out = dataset_dir)
            except:
                print('\nDownload failed. Trying again...')
                eval_urls.append(eval_url)

        ## UNZIP FILES AND DELETE ZIP ARCHIVES
        zipfiles = glob.glob(dataset_dir + os.sep + '*.zip')
        for filepath in zipfiles:
            with zipfile.ZipFile(filepath, 'r') as unzipper:
                unzipper.extractall(dataset_dir)
        for filepath in zipfiles:
            os.remove(filepath)
    else:
        print('All files in dataset verified to exist.')

def get_best_model(eval_features, eval_labels, model_list):
    '''
    Finds best model in a given list of models
    by macro-averaged (class-wise-averaged)
    classification accuracy.
    
    ========
     Inputs
    ========
    
    eval_features : numpy.ndarray
        An array with dimensions (n_samples, n_mels,
        n_bins, n_chs) containing the spectrograms for
        evaluation.
        
    eval_labels : numpy.ndarray
        An array with dimensions (n_samples,
        n_classes) containing the labels for each
        spectrogram (use keras.utils.to_categorical
        if necessary beforehand).
   
    model_list : list of str
        A list of strings, where each string contains
        the path to a .h5 file storing a Keras model
                        
    =========
     Outputs
    =========
    
    output_meta : list of lists
        A list containing lists of three elements:
        The path to a model in model_list, its micro-
        averaged accuracy and its macro-averaged
        accuracy.
        
    best_filepath : str
        The filepath of the model with the highest
        macro-averaged accuracy in model_list.
        
    best_macro_acc : float
        The highest macro-averaged accuracy obtained
        by any of the models in model_list.
    
    best_predictions : numpy.ndarray
        An array with dimensions (n_samples,
        n_classes) containing the predictions by
        class for each sample.

    ==============
     Dependencies
    ==============
    
    numpy (as np), os, tensorflow.keras (as keras),
    glob, tensorflow_model_optimization.sparsity.
    keras (as sparsity), accs (from
    dcase2020task1b_functions)
    '''
    ## INITIALISE SOME VARIABLES
    output_meta = [['model_filepath','micro_acc','macro_acc']]
    best_filepath = ''
    best_macro_acc = 0 # Should be increasing
    best_predictions = np.empty(shape = eval_labels.shape)
    
    ## EXCEPTION HANDLING
    if len(model_list) == 0:
        print('Error: model_list is empty. Please make sure it contains paths to the models you want to test.')
        return output_meta, best_filepath, best_macro_acc, best_predictions

    for model_filepath in model_list:
        ## LOAD MODEL
        with sparsity.prune_scope(): # Need to use this to prevent loading errors
            try:
                model = keras.models.load_model(model_filepath, compile = False) # Don't compile model to save time because we're not training it here.
            except OSError:
                print('Attempted to load ' + model_filepath + ' but file not found. Skipping this model...')
                output_meta.append([model_filepath,0,0])
                continue
        model = sparsity.strip_pruning(model)

        ## MAKE PREDICTIONS
        class_probs = model.predict(eval_features, verbose = False) # (n_samples, n_classes) array with probability of sample being in each class for each element.
        print('Finished predicting for ' + model_filepath)
        
        ## CALCULATE ACCURACIES
        micro_acc, macro_acc = accs(class_probs, eval_labels)

        ## ADD RELEVANT INFORMATION TO OUTPUT
        output_meta.append([model_filepath, micro_acc, macro_acc])
        
        ## COMPARE CURRENT MODEL PERFORMANCE AGAINST CURRENT BEST
        if macro_acc > best_macro_acc:
            best_filepath = model_filepath
            best_macro_acc = macro_acc
            best_predictions = class_probs
    
        ## DELETE RUNNING VARIABLES TO SAVE MEMORY
        try:
            del model, class_probs, micro_acc, macro_acc
        except NameError:
            pass
    
    return output_meta, best_filepath, best_macro_acc, best_predictions

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

import numpy as np
import tensorflow.keras as keras

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

def make_predictions(features, model_list, output_filepath = 'output.csv', class_names = ['indoor', 'outdoor', 'transportation'], filenames = None, save = True, **kwargs):
    '''
    Makes predictions on features by creating
    an ensemble classifier (mean voting) from models
    with filepaths specified in model_list.
    
    ========
     Inputs
    ========
    
    features : numpy.ndarray
        An array with dimensions (n_samples, n_mels,
        n_bins, n_chs) containing the spectrograms for
        test set. It is assumed that features[i]
        corresponds to the audio file i.wav if
        filenames is None and filenames[i] otherwise.
   
    model_list : list of str
        A list of strings, where each string contains
        the path to a .h5 file storing a Keras model.
        All models in this list will be used to make
        an ensemble classifer by averaging their 
        outputs on features. If no ensemble is
        desired, just make sure len(model_list) == 1.
    
    output_filepath : str
        The filepath where the data in output_meta
        will be written to as a .csv file.
        
    class_names : list of str
        A list of strings corresponding to the
        labels for each class predicted by the
        models in model_list.
    
    filenames : list of str or None
        A list of n_samples strings, where the ith
        string corresponds to the raw audio file
        used to generate features[i].
        If None, then it is assumed that the file
        i.wav was used to generate features[i]
        
    save : bool
        If True, saves the data in output_meta to
        a .csv file specified by output_filepath.
        If False, saves nothing regardless of what
        is entered in output_filepath.
    
    **kwargs : dict
        Extra keyword arguments for write_csv.
                        
    =========
     Outputs
    =========
    
    output_meta : list of lists
        A list of lists of 2 + len(class_names)
        elements:
        The name of an audio file, its predicted
        scene label according to the models in
        model_list, and the probabilities of each
        label in class_names

    ==============
     Dependencies
    ==============
    
    numpy (as np), tensorflow.keras (as keras),
    tensorflow_model_optimization.sparsity.keras
    (as sparsity)
    '''
    header = class_names.copy()
    header.insert(0, 'scene_label')
    header.insert(0, 'filename')
    
    ## INITIALISE SOME VARIABLES
    output_meta = [header]
    
    ## EXCEPTION HANDLING
    if len(model_list) == 0:
        print('Error: model_list is empty. Please make sure it contains paths to the models you want to test.')
        return output_meta
    
    if filenames is None:
        filenames = ['{}.wav'.format(i) for i in range(len(features))]
    elif len(filenames) != len(features):
        print('Error: filenames does not have same length as features. Please make sure each element in features corresponds to a file name.')
        return output_meta
    
    ## MAKE ENSEMBLE PREDICTIONS
    class_probs = np.zeros(shape = (len(features), len(class_names)))
    n_models = 0
    
    for model_filepath in model_list:       
        try:
            ## LOAD MODEL
            with sparsity.prune_scope(): # Need to use this to prevent loading errors
                model = keras.models.load_model(model_filepath, compile = False) # Don't compile model to save time because we're not training it here.
            model = sparsity.strip_pruning(model)
            
            ## MAKE SINGLE MODEL PREDICTIONS
            class_probs += model.predict(features, verbose = False) # (n_samples, n_classes) array with probability of sample being in each class for each element.
            n_models += 1
            print('Finished making predictions for ' + model_filepath)
            
            ## DELETE RUNNING VARIABLES TO SAVE MEMORY
            try:
                del model
            except NameError:
                pass
        except OSError:
            print('Attempted to load ' + model_filepath + ' but file not found. Skipping this model from ensemble...')
            continue
        except:
            print('Attempted to make predictions for ' + model_filepath + ' but failed. Skipping this model from ensemble...')
            continue
    
    class_probs /= n_models # Average out predictions
        
    ## ADD RELEVANT INFORMATION TO OUTPUT
    for i, filename in enumerate(filenames):
        row = list(class_probs[i])
        row.insert(0, class_names[np.argmax(class_probs[i])])
        row.insert(0, filename)
        output_meta.append(row)
    
    if save:
        write_csv(output_meta, filepath = output_filepath, **kwargs)

    return output_meta

def read_csv(filepath = '', delimiter = ',', list_of_str = False, **kwargs):
    '''
    Reads .csv files and outputs them as a list of
    (lists of) strings.
    
    ========
     Inputs
    ========
    
    filepath : str
        The path to the .csv file to be read.
        
    delimiter : char
        The delimiter used to separate entries in the
        same row in the .csv file.
        
    list_of_str : bool
        If True, data will be read as though it were a
        list of strings, i.e. as though the .csv file
        contains only one item per row. Otherwise,
        data will be read as though it were 
        a list of lists, i.e. as though the .csv file
        contains more than one item per row.
        It is usually advisable to set this to "True"
        if the .csv file to be read contains only a
        single column.

    **kwargs : dict
        Other keyword arguments to pass to the Python
        function open OTHER than 'mode'.
        
    =========
     Outputs
    =========
    
    data : list of lists of str or list of str
        The data read from the .csv file. If
        list_of_str is True, then data[i][j] contains
        the j-th element of the i-th row in the .csv
        file as a string. Otherwise, data[i] contains
        the text in the i-th row in the .csv file.
        
    ==============
     Dependencies
    ==============
    
    csv
    
    =======
     Notes
    =======
    
    Sometimes, depending on the file format, we need
    to add newline = '\n' as an argument in **kwargs
    for the file to be read correctly.
    
    Written by Kenneth Ooi
    '''
    data = []
    with open(filepath, mode = 'r', **kwargs) as csvfile:
        my_csv_reader = csv.reader(csvfile, dialect = 'excel', delimiter = delimiter)
        if list_of_str:
            for line in my_csv_reader:
                data.append(line[0])
        else:
            for line in my_csv_reader:
                data.append(line)
    return data

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
    
    =======
     Notes
    =======
    If using TensorFlow 1.x, the variable names
    for tracking metrics are 'val_acc' and 'acc'
    instead of 'val_accuracy' and 'accuracy'.
    Change the key values in the formatting
    strings accordingly to avoid a key error when
    running in TF 1.x.
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

def write_csv(data, filepath = '', delimiter = ',', list_of_str = False, mode = 'w', **kwargs):
    '''
    Writes a list of lists to a .csv file.
    
    ========
     Inputs
    ========
    
    data : list of lists or list of str
        The data to be written to the .csv file. If a
        list of lists, then data[i][j] should contain
        either a string or an object that can be
        typecasted to a string, and will be written as
        the j-th element of the i-th row in the .csv
        file. If a list of strings, then data[i]
        should contain the string to be written to the
        i-th row of the .csv file.
        
    filepath : str
        The path to write the .csv file to.
        
    delimiter : char
        The delimiter used to separate entries in the
        same row in the .csv file.
    
    list_of_str : bool
        If True, data will be assumed to be a list of
        strings. Otherwise, data will be assumed to be
        a list of lists.
    
    mode : str
        If 'w', will overwrite the file at filepath if
        it already exists. If 'x', will throw an error
        if the file at filepath already exists.
    
    **kwargs : dict
        Other keyword arguments to pass to the Python
        function open OTHER than 'mode'.
    
    =========
     Outputs
    =========
    
    None. However, a .csv file will be written to the
    specified filepath.
    
    ==============
     Dependencies
    ==============
    
    csv
    
    ==========
     Examples
    ==========
    
    # Write a list of strings to a .csv file (Windows)
    >>> my_list = ['alpha','bravo','charlie','delta']
    >>> write_csv(my_list, filepath = 'my_list.csv', list_of_str = True, newline = '\n')
    
    Written by Kenneth Ooi
    '''
    with open(filepath, mode = mode, **kwargs) as csvfile: # Save the metadata using csv.writer.
        my_csv_writer = csv.writer(csvfile, delimiter = delimiter, dialect='excel')
        if list_of_str:
            for row in data:
                my_csv_writer.writerow([row])
        else:
            my_csv_writer.writerows(data)
