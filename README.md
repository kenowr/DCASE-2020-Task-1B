# DCASE-2020-Task-1B
Source code for the submission to DCASE 2020 Task 1B

(Files have been uploaded, README is under construction)

Run `model1.py`, `model2.py`, `model3.py`, and `model4.py` to obtain the reported results in the technical report.

The conda environment used to run the functions is given in `dcase2020task1b.yml`.

The `modelx.py` files will auto-download the dataset from Zenodo if it has not yet been downloaded, but since the development and evaluation sets are relatively large (38.6GB and 23.1GB respectively), you may want to use your own download manager to download the datasets before running the code in this repository. In that case, please place the unzipped files from Zenodo into the project repository in the following structure:
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
