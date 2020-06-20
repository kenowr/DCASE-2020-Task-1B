# DCASE-2020-Task-1B

Source code for the submission to DCASE 2020 Task 1B (Kenneth Ooi, Santi Peksi, Woon-Seng Gan)


(Files have been uploaded, README is under construction)

Getting started
---------------




Run `model1.py`, `model2.py`, `model3.py`, and `model4.py` to obtain the reported results in the technical report.

The conda environment used to run the functions is given in `dcase2020task1b.yml`. See <a href='#system_setup'> here</a>.

The `model1.py`, `model2.py`, `model3.py`, and `model4.py` files will auto-download the dataset from Zenodo if it has not yet been downloaded, but since the development and evaluation sets are relatively large (38.6GB and 23.1GB, downloadable from https://zenodo.org/record/3670185 and https://zenodo.org/record/3685835 respectively), you may want to use your own download manager to download the datasets before running the code in this repository. In that case, please place the unzipped files from Zenodo into the project repository in the following structure:

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
    ┆                                                              
    ┆                                                              # Other files in this repository
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
- conda version 4.8.3 (downloadable <a href='https://www.anaconda.com/products/individual'> here </a>)
- Python version 3.7.6
- TensorFlow version 2.1.0

The exact conda environment that we used to test the code in this repository is given in the file `dcase2020task1b.yml`, which can be cloned, for example, by entering `conda env create -f dcase2020task1b.yml` into an Anaconda prompt terminal.
