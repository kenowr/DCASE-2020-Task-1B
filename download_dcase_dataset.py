import zipfile, os, wget, glob
from read_csv import read_csv

__version__ = '20200615_dcase2020'

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