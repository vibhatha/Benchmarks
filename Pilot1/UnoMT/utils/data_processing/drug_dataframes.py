""" 
    File Name:          UnoPytorch/drug_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:   
        This file takes care of all the dataframes related drug features.
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
import pyarrow as pa
from pyarrow import csv
from pyarrow.csv import ReadOptions
from pyarrow.csv import ConvertOptions
from pyarrow.csv import ParseOptions
import time

ctx = CylonContext(config=None, distributed=False)

# pycylon imports end

logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = './raw/'
PROC_FOLDER = './processed/'

# All the filenames related to the drug features
ECFP_FILENAME = 'pan_drugs_dragon7_ECFP.tsv'
PFP_FILENAME = 'pan_drugs_dragon7_PFP.tsv'
DSCPTR_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'
# Drug property file. Does not exist on FTP server.
DRUG_PROP_FILENAME = 'combined.panther.targets'

# Use only the following target families for classification
TGT_FAMS = ['transferase', 'oxidoreductase', 'signaling molecule',
            'nucleic acid binding', 'enzyme modulator', 'hydrolase',
            'receptor', 'transporter', 'transcription factor', 'chaperone']


def get_drug_fgpt_df(data_root: str,
                     int_dtype: type = np.int8):
    """df = get_drug_fgpt_df('./data/')

    This function loads two drug fingerprint files, join them as one
    dataframe, convert them to int_dtype and return.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug fingerprint dataframe.
    """
    print("=" * 80)
    t_start = time.time()
    print("get_drug_fgpt_df")
    df_filename = 'drug_fgpt_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    print(f">> df_filename : {df_filename}")

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
        print(f"Reading from pickle : {df_path}")
        print("DataFrame:>>>")
        print(df)
        print(df.index)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug fingerprint dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=[ECFP_FILENAME, PFP_FILENAME],
                       target_folder=os.path.join(data_root, RAW_FOLDER))
        print(f"ecfp_df file: {os.path.join(data_root, RAW_FOLDER, ECFP_FILENAME)}")
        t_s_1_load = time.time()
        # ecfp_df = pd.read_csv(
        #     os.path.join(data_root, RAW_FOLDER, ECFP_FILENAME),
        #     sep='\t',
        #     #header=None,
        #     #index_col=0,
        #     skiprows=[0, ])
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t").skip_rows(1)
        ecfp_tb: Table = read_csv(ctx, os.path.join(data_root, RAW_FOLDER, ECFP_FILENAME),
                                  csv_read_options)
        t_e_1_load = time.time()
        col_names = [str(i) for i in range(0, ecfp_tb.shape[1])]
        # ecfp_df.columns = col_names
        ecfp_tb.rename(col_names)
        ecfp_tb = ecfp_tb.add_prefix("1_")
        # ecfp_df = ecfp_df.add_prefix("1_")
        # ecfp_df.set_index(ecfp_df.columns[0], drop=True, inplace=True)
        ecfp_tb.set_index(ecfp_tb.column_names[0], drop=True)
        print("Head of ecfp_df")
        # assert ecfp_df.index.values.tolist() == ecfp_tb.index.values.tolist()

        print(f"DataFrame: ecfp_df = {ecfp_tb.shape}")
        print(f"Data Loadding ecfp_df: {t_e_1_load - t_s_1_load} s")

        print(f"pfp_df file: {os.path.join(data_root, RAW_FOLDER, PFP_FILENAME)}")
        t_s_2_load = time.time()
        # pfp_df = pd.read_csv(
        #     os.path.join(data_root, RAW_FOLDER, PFP_FILENAME),
        #     sep='\t',
        #     #header=None,
        #     #index_col=0,
        #     skiprows=[0, ])
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t").skip_rows(1)
        pfp_tb: Table = read_csv(ctx, os.path.join(data_root, RAW_FOLDER, PFP_FILENAME),
                                 csv_read_options)
        t_e_2_load = time.time()
        col_names = [str(i) for i in range(0, pfp_tb.shape[1])]
        # pfp_df.columns = col_names
        pfp_tb.rename(col_names)
        pfp_tb = pfp_tb.add_prefix("2_")
        # pfp_df = pfp_df.add_prefix("2_")
        # pfp_df.set_index(pfp_df.columns[0], drop=True, inplace=True)
        pfp_tb.set_index(pfp_tb.column_names[0], drop=True)
        print(f"DataFrame: pfp_df = {pfp_tb.shape}")
        print(f"Data Loadding pfp_df: {t_e_2_load - t_s_2_load} s")
        print("Head of pfp_df")

        # assert pfp_df.index.values.tolist() == pfp_tb.index.values.tolist()

        # print(ecfp_df.index.values.tolist()[0:5], ecfp_tb.index.values.tolist()[0:5])

        # print(ecfp_df.index.values.tolist()[0:5], ecfp_tb.index.values.tolist()[0:5])
        t_concat = time.time()
        # df = pd.concat([ecfp_df, pfp_df], axis=1, join='inner')
        tb = Table.concat([ecfp_tb, pfp_tb], axis=1, join='inner')
        t_concat = time.time() - t_concat
        tb = tb.astype(int)
        # print("idx_names: ", df.index.names)
        # print(ecfp_df.index.values.tolist()[0:5], ecfp_tb.index.values.tolist()[0:5])
        # print(df.index.values.tolist()[0:5], tb.index.values.tolist()[0:5])
        # assert df.index.values.tolist().sort() == tb.index.values.tolist().sort()
        #tb.reset_index()
        #df = tb.to_pandas()
        #df.set_index(df.columns[0], inplace=True)
        print(f"Concat Time : {t_concat} s")
        #print(f"Concatenated Pdf : {df.shape} {tb.shape}")
        #print(df)
        #print("idx_names: ", df.index.names)

        # Convert data type into generic python types
        # df = df.astype(int)

        #print(f"DataFrame Types: {df.dtypes}")

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
            print(f"Save Path : {os.path.join(data_root, PROC_FOLDER)}")
        except FileExistsError:
            pass
        # avoid saving to pickle to run the whole data processing step
        # df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    #df = df.astype(int_dtype)
    tb = tb.astype('int8')
    t_end = time.time()
    print(f"Total time taken get_drug_fgpt_df : {t_end - t_start} s")
    print("=" * 80)
    return tb


def get_drug_dscptr_df(data_root: str,
                       dscptr_scaling: str,
                       dscptr_nan_thresh: float,
                       float_dtype: type = np.float32):
    """df = get_drug_dscptr_df('./data/', 'std', 0.0)

    This function loads the drug descriptor file, process it and return
    as a dataframe. The processing includes:
        * removing columns (features) and rows (drugs) that have exceeding
            ratio of NaN values comparing to nan_thresh;
        * scaling all the descriptor features accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        dscptr_scaling (str): scaling strategy for all descriptor features.
        dscptr_nan_thresh (float): threshold ratio of NaN values.
        float_dtype (float): float dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug descriptor dataframe.
    """
    print("=" * 80)
    t_start = time.time()
    print("get_drug_dscptr_df")
    df_filename = 'drug_dscptr_df(scaling=%s, nan_thresh=%.2f).pkl' \
                  % (dscptr_scaling, dscptr_nan_thresh)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    print(f"File Name: {df_filename}")
    print(f"Df Path: {df_path}")

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        print(f"Reading from pickle {df_path}")
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        print("Reading from scratch")
        logger.debug('Processing drug descriptor dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=DSCPTR_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))
        print(f"Data File path: {os.path.join(data_root, RAW_FOLDER, DSCPTR_FILENAME)}")
        t_s_load = time.time()
        # df = pd.read_csv(
        #     os.path.join(data_root, RAW_FOLDER, DSCPTR_FILENAME),
        #     sep='\t',
        #     #header=0,
        #     # index_col=0, # DON"T DO INDEXING HERE
        #     na_values='na')
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t").na_values(['na'])
        tb: Table = read_csv(ctx, os.path.join(data_root, RAW_FOLDER, DSCPTR_FILENAME),
                             csv_read_options)
        t_e_load = time.time()

        print(f"Data Loading time : {t_e_load - t_s_load} s")
        print(f"DataFrame : {tb.shape}")

        # Drop NaN values if the percentage of NaN exceeds nan_threshold
        # Note that columns (features) are dropped first, and then rows (drugs)
        valid_thresh = 1.0 - dscptr_nan_thresh
        # df.dropna(axis=1, inplace=True, thresh=int(df.shape[0] * valid_thresh))
        tb.dropna(axis=0, inplace=True)
        # df.dropna(axis=0, inplace=True, thresh=int(df.shape[1] * valid_thresh))
        tb.dropna(axis=1, inplace=True)

        df = tb.to_pandas()
        df.set_index(df.columns[0], inplace=True)

        # Fill the rest of NaN with column means
        df.fillna(df.mean(), inplace=True)
        # TODO: fix this to work with list of values or a table with one row
        # tb = tb.fillna(np.mean(df.mean()))
        # df = tb.to_pandas()

        # Scaling the descriptor with given scaling method
        df = scale_dataframe(df, dscptr_scaling)

        tb = Table.from_pandas(ctx, df, preserve_index=True)
        tb.set_index(tb.column_names[-1], drop=True)

        # Convert data type into generic python types
        #df = df.astype(float)
        tb = tb.astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        # df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    #df = df.astype(float_dtype)
    tb = tb.astype('float32')
    # print(f">>>> df.index : {df.index}")
    t_end = time.time()
    print(f"Total Time taken get_drug_dscptr_df: {t_end - t_start} s")
    print("=" * 80)
    return tb


def get_drug_feature_df(data_root: str,
                        drug_feature_usage: str,
                        dscptr_scaling: str,
                        dscptr_nan_thresh: float,
                        int_dtype: type = np.int8,
                        float_dtype: type = np.float32):
    """df = get_drug_feature_df('./data/', 'both', 'std', 0.0)

    This function utilizes get_drug_fgpt_df and get_drug_dscptr_df. If the
    feature usage is 'both', it will loads fingerprint and descriptors,
    join them and return. Otherwise, if feature usage is set to
    'fingerprint' or 'descriptor', the function returns the corresponding
    dataframe.

    Args:
        data_root (str): path to the data root folder.
        drug_feature_usage (str): feature usage indicator. Choose between
            'both', 'fingerprint', and 'descriptor'.
        dscptr_scaling (str): scaling strategy for all descriptor features.
        dscptr_nan_thresh (float): threshold ratio of NaN values.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug feature dataframe.
    """
    print("=" * 80)
    print("get_drug_feature_df")
    # Return the corresponding drug feature dataframe
    if drug_feature_usage == 'both':
        print(f"drug_feature_usage: {drug_feature_usage}")
        print("=" * 80)
        drug_fgpt_tb = get_drug_fgpt_df(data_root=data_root,
                                        int_dtype=int_dtype)
        drug_fgpt_tb.reset_index()
        drug_fgpt_df = drug_fgpt_tb.to_pandas()
        drug_fgpt_df.set_index(drug_fgpt_df.columns[0], drop=True, inplace=True)
        drug_fgpt_tb.set_index(drug_fgpt_tb.column_names[0], drop=True)

        drug_dscptr_tb = get_drug_dscptr_df(data_root=data_root,
                                            dscptr_scaling=dscptr_scaling,
                                            dscptr_nan_thresh=dscptr_nan_thresh,
                                            float_dtype=float_dtype)
        drug_dscptr_tb.reset_index()
        drug_dscptr_df = drug_dscptr_tb.to_pandas()
        drug_dscptr_df.set_index(drug_dscptr_df.columns[0], drop=True, inplace=True)
        drug_dscptr_tb.set_index(drug_dscptr_tb.column_names[0], drop=True)
        print(f">>>> get_drug_feature_df.Concat Columns: {len(drug_fgpt_df.columns)}, "
              f"{len(drug_dscptr_df.columns)}")
        t_concat = time.time()
        #concat_df = pd.concat([drug_fgpt_df, drug_dscptr_df], axis=1, join='inner')
        concat_tb = Table.concat([drug_fgpt_tb, drug_dscptr_tb], axis=1, join='inner')
        t_concat = time.time() - t_concat
        print(f"Concat Time [drug_feature_usage={drug_feature_usage}] : {t_concat} s, "
              f"shape[{drug_fgpt_df.shape}, {drug_dscptr_df.shape}]")
        return concat_tb
    elif drug_feature_usage == 'fingerprint':
        print(f"drug_feature_usage: {drug_feature_usage}")
        return get_drug_fgpt_df(data_root=data_root,
                                int_dtype=int_dtype)
    elif drug_feature_usage == 'descriptor':
        print(f"drug_feature_usage: {drug_feature_usage}")
        return get_drug_dscptr_df(data_root=data_root,
                                  dscptr_scaling=dscptr_scaling,
                                  dscptr_nan_thresh=dscptr_nan_thresh,
                                  float_dtype=float_dtype)
    else:
        print(f"drug_feature_usage: {drug_feature_usage}")
        logger.error('Drug feature must be one of \'fingerprint\', '
                     '\'descriptor\', or \'both\'.', exc_info=True)
        raise ValueError('Undefined drug feature %s.' % drug_feature_usage)


def get_drug_prop_df(data_root: str):
    """df = get_drug_prop_df('./data/')

    This function loads the drug property file and returns a dataframe with
    only weighted QED and target families as columns (['QED', 'TARGET']).

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.

    Returns:
        pd.DataFrame: drug property dataframe with target families and QED.
    """
    print("=" * 80)
    df_filename = 'drug_prop_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    print(f"DataFrame File : {df_filename}")
    print(f"DF Path : {df_path}")

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        print(f"Reading from pickle: {df_path}")
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        print("Reading from scratch")
        logger.debug('Processing drug targets dataframe ... ')

        raw_file_path = os.path.join(data_root, RAW_FOLDER, DRUG_PROP_FILENAME)
        print(f"raw_file_path : {raw_file_path}")

        # Download the raw file if not exist
        download_files(filenames=DRUG_PROP_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))
        use_cols = ['anl_cpd_id', 'qed_weighted', 'target_families']
        # df = pd.read_csv(
        #     raw_file_path,
        #     sep='\t',
        #     header=0,
        #     index_col=0,
        #     usecols=use_cols)

        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t").use_cols(use_cols)
        tb: Table = read_csv(ctx, raw_file_path, csv_read_options)

        # print(f"DataFrame : {df.shape}")
        # print("Index: ")
        # print(df.index)

        # Change index name for consistency across the whole program
        # df.index.names = ['DRUG_ID']
        # df.columns = ['QED', 'TARGET', ]
        new_column_names = ['DRUG_ID', 'QED', 'TARGET']
        tb.rename(new_column_names)

        # print(f"df index names: {df.index.names}")
        # print(f"df.columns: {df.columns}")

        # Convert data type into generic python types
        # df[['QED']] = df[['QED']].astype(float)
        # df[['TARGET']] = df[['TARGET']].astype(str)
        # tb[['QED']] = tb[['QED']].astype(float)
        # tb[['TARGET']] = tb[['TARGET']].astype(str)
        tb = tb[tb['QED'] != '']
        tb.set_index(tb.column_names[0], drop=True)
        tb = tb.astype({'QED': float, 'TARGET': str})
        print(tb.to_arrow(), tb.shape)
        print(tb)
        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        # df.to_pickle(df_path)
    print("=" * 80)
    tb.reset_index()
    df = tb.to_pandas()
    df.set_index(df.columns[0], inplace=True, drop=True)
    return df


def get_drug_target_df(data_root: str,
                       int_dtype: type = np.int8):
    """df = get_drug_target_df('./data/')

    This function the drug property dataframe, process it and return the
    drug target families dataframe. The processing includes:
        * removing all columns but 'TARGET';
        * drop drugs/rows that are not in the TGT_FAMS list;
        * encode target families into integer labels;
        * convert data types for more compact structure;

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug target families dataframe.
    """
    print("=" * 80)
    print("get_drug_target_df")
    df = get_drug_prop_df(data_root=data_root)[['TARGET']]

    # Only take the rows with specific target families for classification
    df = df[df['TARGET'].isin(TGT_FAMS)][['TARGET']]
    # tb = tb[tb['TARGET'].isin(TGT_FAMS)][['TARGET']]

    # print(f"Check IsIn {df.shape}, {tb.shape}")
    # df = tb.to_pandas()
    # assert df.values.flatten().tolist() == tb.to_pandas().values.flatten().tolist()

    # Encode str formatted target families into integers
    df['TARGET'] = encode_label_to_int(data_root=data_root,
                                       dict_name='drug_target_dict.txt',
                                       labels=df['TARGET'])

    # Convert the dtypes for a more efficient, compact dataframe
    # Note that it is safe to use int8 here for there are only 10 classes
    print("=" * 80)
    return df.astype(int_dtype)


def get_drug_qed_df(data_root: str,
                    qed_scaling: str,
                    float_dtype: type = np.float32):
    """df = get_drug_qed_df('./data/', 'none')


    This function the drug property dataframe, process it and return the
    drug weighted QED dataframe. The processing includes:
        * removing all columns but 'QED';
        * drop drugs/rows that have NaN as weighted QED;
        * scaling the QED accordingly;
        * convert data types for more compact structure;

    Args:
        data_root (str): path to the data root folder.
        qed_scaling (str): scaling strategy for weighted QED.
        float_dtype (float): float dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug weighted QED dataframe.
    """
    print("=" * 80)
    print("get_drug_qed_df")
    df = get_drug_prop_df(data_root=data_root)[['QED']]
    tb = Table.from_pandas(ctx, df)
    print(f">>> Table Shape Before Dropna {tb.shape} , {df.shape}")
    # Drop all the NaN values before scaling
    df.dropna(axis=0, inplace=True)
    # tb.dropna(axis=1, inplace=True) # TODO:: issue handle
    print(f">>> get_drug_qed_df.DropNa : {df.shape} , {tb.shape}")

    # Note that weighted QED is by default already in the range of [0, 1]
    # Scaling the weighted QED with given scaling method
    # df = tb.to_pandas()
    df = scale_dataframe(df, qed_scaling)

    # Convert the dtypes for a more efficient, compact dataframe
    print("=" * 80)
    return df.astype(float_dtype)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    print('=' * 80 + '\nDrug feature dataframe head:')
    drug_feature_tb = get_drug_feature_df(data_root='../../data/',
                              drug_feature_usage='both',
                              dscptr_scaling='std',
                              dscptr_nan_thresh=0.)
    drug_feature_df = drug_feature_tb.to_pandas()
    print(drug_feature_df.head())

    # print('=' * 80 + '\nDrug target families dataframe head:')
    # print(get_drug_target_df(data_root='../../data/').head())
    #
    # print('=' * 80 + '\nDrug target families dataframe head:')
    # print(get_drug_qed_df(data_root='../../data/', qed_scaling='none').head())
