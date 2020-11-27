# DCASE-2020-Task-1B

Source code for the submission to DCASE 2020 Task 1B (Kenneth Ooi, Santi Peksi, Woon-Seng Gan)

Getting started
---------------
Firstly, clone this repository by manually downloading it from https://github.com/kenowr/DCASE-2020-Task-1B.git, or enter the following line from a terminal:

    git clone https://github.com/kenowr/DCASE-2020-Task-1B.git

To train and print metrics for all models in the technical report, simply run the following line from a terminal:

    python main.py
    
To train and print metrics for only specific models, specify the model number(s) after the function call to `main.py`. For example, to train Model 4 followed by Model 1, enter the following:

    python main.py 4 1

Alternatively, the individual model files `model1.py`, `model2.py`, `model3.py`, and `model4.py` can also be run to obtain the reported results in the technical report. For example, running `python model3.py` has the same effect as running `python main.py 3`.

However, avoid running the same models multiple times without clearing the directories where files are saved to each time.

The specific system setup that we used to obtain the results for the technical paper is described in the <a href='#system_setup'>following section</a>, with the exact conda environment being given in the file `dcase2020task1b.yml` found in this repository.

Dataset
-------
The dataset we used to train the models for our submission is the TAU Urban Acoustic Scenes 2020 3Class dataset. The files `main.py`, `model1.py`, `model2.py`, `model3.py`, and `model4.py` will automatically download the dataset from Zenodo if it has not yet been downloaded, but since the development and evaluation sets are relatively large (38.6GB and 23.1GB, downloadable from https://zenodo.org/record/3670185 and https://zenodo.org/record/3685835 respectively), you may want to use your own download manager to download the datasets before running the code in this repository. In that case, please place the unzipped files from Zenodo into the project repository such that the following directory structure is matched:

    .
    ├── datasets                                                   # Dataset files from Zenodo
    │   ├── TAU-urban-acoustic-scenes-2020-3class-development      # Development set
    │   │   ├── audio
    │   │   ├── evaluation_setup
    │   │   ├── LICENSE
    │   │   ├── README.html
    │   │   └── README.md
    │   │
    │   └── TAU-urban-acoustic-scenes-2020-3class-evaluation       # Evaluation set
    │       ├── audio
    │       ├── evaluation_setup
    │       ├── LICENSE
    │       ├── meta.csv
    │       ├── README.html
    │       └── README.md
    │   
    ├── Ooi_NTU_task1b.technical_report.pdf                        # Technical report
    ├── README.md                                                  # This file
    ├── dataset_filenames.csv                                      # List of filenames in dataset (for validation)
    ├── dcase2020task1b.yml                                        # conda environment file for this repository
    ├── dcase2020task1b_functions.py                               # Functions called by model1.py, ..., model4.py
    ├── devt_urls.csv                                              # URLs to download the development set from
    ├── eval_urls.csv                                              # URLs to download the evaluation set from
    ├── main.py
    ├── model1.py
    ├── model2.py
    ├── model3.py
    ├── model4.py
    └── model_size_calculation.py                                  # Last file in this repository in lexicographic order
    
System setup <a name='system_setup'>
------------
The system we used to test the code in this repository is as follows:
- OS: Windows 10
- Processor: Intel Core i9-7900X CPU @ 3.30GHz
- RAM: 64GB
- GPU: NVIDIA GeForce GTX 1080 Ti

The main driver and software versions were:
- CUDA version 10.1 (downloadable <a href='https://developer.nvidia.com/cuda-10.1-download-archive-base'> here </a> with an NVIDIA account)
- cuDNN version 7.6.5 (downloadable <a href='https://developer.nvidia.com/rdp/cudnn-download'> here </a> with an NVIDIA account)
- conda version 4.8.3 (downloadable <a href='https://www.anaconda.com/products/individual'> here</a>)
- Python version 3.7.6
- TensorFlow version 2.1.0

The exact conda environment that we used to test the code in this repository is given in the file `dcase2020task1b.yml`, which can be cloned, for example, by entering `conda env create -f dcase2020task1b.yml` into an Anaconda prompt terminal.

Contact
-------
If you encounter any problems/bugs in the source code, please drop an email at <a href='mailto:wooi002@e.ntu.edu.sg'> wooi002@e.ntu.edu.sg</a>.
