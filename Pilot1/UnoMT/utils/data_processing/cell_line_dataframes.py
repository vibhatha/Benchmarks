""" 
    File Name:          UnoPytorch/cell_line_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:   
        This file takes care of all the dataframes related cell lines.
"""
import os
import logging

import numpy as np
import pandas as pd

from utils.data_processing.dataframe_scaling import scale_dataframe
from utils.data_processing.label_encoding import encode_label_to_int
from utils.miscellaneous.file_downloading import download_files

# pycylon imports start

from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import time

ctx = CylonContext(config=None, distributed=False)

# pycylon imports end

logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = './raw/'
PROC_FOLDER = './processed/'

# All the filenames related to cell lines
CL_METADATA_FILENAME = 'combined_cl_metadata'
RNASEQ_SOURCE_SCALE_FILENAME = 'combined_rnaseq_data_lincs1000_source_scale'
RNASEQ_COMBAT_FILENAME = 'combined_rnaseq_data_lincs1000_combat'


def get_rna_seq_df(data_root: str,
                   rnaseq_feature_usage: str,
                   rnaseq_scaling: str,
                   float_dtype: type = np.float32):
    """df = get_rna_seq_df('./data/', 'source_scale', 'std')

    This function loads the RNA sequence file, process it and return
    as a dataframe. The processing includes:
        * remove the '-' in cell line names;
        * remove duplicate indices;
        * scaling all the sequence features accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        rnaseq_feature_usage (str): feature usage indicator, Choose between
            'source_scale' and 'combat'.
        rnaseq_scaling (str): scaling strategy for RNA sequence.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed RNA sequence dataframe.
    """
    t_start = time.time()
    print("========>>> get_rna_seq_df")
    df_filename = 'rnaseq_df(%s, scaling=%s).pkl' \
                  % (rnaseq_feature_usage, rnaseq_scaling)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        print("Reading From Pickle")
        print(f"DF PATH: {df_path}/{df_filename}")
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        print("Loading from scratch...")
        logger.debug('Processing RNA sequence dataframe ... ')

        if rnaseq_feature_usage == 'source_scale':
            raw_data_filename = RNASEQ_SOURCE_SCALE_FILENAME
        elif rnaseq_feature_usage == 'combat':
            raw_data_filename = RNASEQ_COMBAT_FILENAME
        else:
            logger.error('Unknown RNA feature %s.' % rnaseq_feature_usage,
                         exc_info=True)
            raise ValueError('RNA feature usage must be one of '
                             '\'source_scale\' or \'combat\'.')

        # Download the raw file if not exist
        download_files(filenames=raw_data_filename,
                       target_folder=os.path.join(data_root, RAW_FOLDER))
        print("cell_file: ", os.path.join(data_root, RAW_FOLDER, raw_data_filename))
        t_s_load = time.time()
        # df = pd.read_csv(
        #     os.path.join(data_root, RAW_FOLDER, raw_data_filename),
        #     sep='\t',
        #     header=0,
        #     index_col=0)
        #t_e_load = time.time()
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t")

        tb: Table = read_csv(ctx, os.path.join(data_root, RAW_FOLDER, raw_data_filename), csv_read_options)
        t_e_load = time.time()
        print(f"Cylon Shape Before Duplicate Removal: {tb.shape}")
        tb = tb.unique([tb.column_names[0]], keep='first')
        print(f"Cylon Shape After Duplicate Removal: {tb.shape}")
        df = tb.to_pandas()
        df = df.set_index(df.columns[0])
        # # Delete '-', which could be inconsistent between seq and meta
        df.index = df.index.str.replace('-', '')

        # Note that after this name changing, some rows will have the same
        # name like 'GDSC.TT' and 'GDSC.T-T', but they are actually the same
        # Drop the duplicates for consistency
        print(f"Pandas Shape Before Duplicate Removal: {df.shape}")
        df = df[~df.index.duplicated(keep='first')]
        print(f"Pandas Shape After Duplicate Removal: {df.shape}")

        # Scaling the descriptor with given scaling method
        df = scale_dataframe(df, rnaseq_scaling)

        # Convert data type into generic python types
        df = df.astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        # Commenting to avoid loading from cache
        #df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df = df.astype(float_dtype)
    t_end = time.time()
    print(f"Data Loading time : {t_e_load - t_s_load}")
    print(f"Total time for get_rna_seq_df : {t_end - t_start} s")
    return df


def get_cl_meta_df(data_root: str,
                   int_dtype: type = np.int8):
    """df = get_cl_meta_df('./data/')

    This function loads the metadata for cell lines, process it and return
    as a dataframe. The processing includes:
        * change column names to ['data_src', 'site', 'type', 'category'];
        * remove the '-' in cell line names;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed cell line metadata dataframe.
    """
    t_start = time.time()

    print("=" * 80)
    print("========>>> get_cl_meta_df")
    df_filename = 'cl_meta_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        print(f"Reading from Path: {df_path}")
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        print("Loading from scratch...")
        logger.debug('Processing cell line meta dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=CL_METADATA_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))
        use_cols = ['sample_name',
                     'dataset',
                     'simplified_tumor_site',
                     'simplified_tumor_type',
                     'sample_category']
        # t_s_load = time.time()
        # df = pd.read_csv(
        #     os.path.join(data_root, RAW_FOLDER, CL_METADATA_FILENAME),
        #     sep='\t',
        #     header=0,
        #     index_col=0,
        #     usecols=use_cols,
        #     dtype=str)
        # t_e_load = time.time()
        #
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t")
        t_s_load = time.time()
        tb: Table = read_csv(ctx, os.path.join(data_root, RAW_FOLDER, CL_METADATA_FILENAME),
                             csv_read_options)
        tb = tb[use_cols]
        t_e_load = time.time()
        df = tb.to_pandas()
        df = df.set_index(df.columns[0])
        # Renaming columns for shorter and better column names
        print(f"DataFrame shape {df.shape}")
        print(f"Data Loading time : {t_e_load - t_s_load}")
        df.index.names = ['sample']
        df.columns = ['data_src', 'site', 'type', 'category']

        print("Cell DataFrame: ")
        print(df)

        # Delete '-', which could be inconsistent between seq and meta
        print(f"Before replacing str: {df.shape}")
        df.index = df.index.str.replace('-', '')
        print(f"After replacing str: {df.shape}")

        # Convert all the categorical data from text to numeric
        columns = df.columns
        dict_names = [i + '_dict.txt' for i in columns]
        for col, dict_name in zip(columns, dict_names):
            df[col] = encode_label_to_int(data_root=data_root,
                                          dict_name=dict_name,
                                          labels=df[col])

        # Convert data type into generic python types
        df = df.astype(int)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
            print(f"Save Folder : {os.path.join(data_root, PROC_FOLDER)}" )
        except FileExistsError:
            pass
        # avoid loading from pickle to test cylon integration
        #df.to_pickle(df_path)

    df = df.astype(int_dtype)
    t_end = time.time()
    print(f"Total time taken get_cl_meta_df : {t_end - t_start} s")
    print("=" * 80)
    return df


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print('=' * 80 + '\nRNA sequence dataframe head:')
    print(get_rna_seq_df(data_root='../../data/',
                         rnaseq_feature_usage='source_scale',
                         rnaseq_scaling='std').head())

    print('=' * 80 + '\nCell line metadata dataframe head:')
    print(get_cl_meta_df(data_root='../../data/').head())
