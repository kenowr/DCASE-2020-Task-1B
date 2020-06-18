import csv

__version__ = '20200615_dcase2020'

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