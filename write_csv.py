import csv

__version__ = '20200615_dcase2020'

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