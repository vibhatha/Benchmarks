""" 
    File Name:          UnoPytorch/response_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:   
        This file takes care of all the dataframes related drug response.
"""


import os
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import multiprocessing
from joblib import Parallel, delayed

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

# All the filenames related to the drug response
DRUG_RESP_FILENAME = 'rescaled_combined_single_drug_growth'


def get_drug_resp_df(data_root: str,
                     grth_scaling: str,
                     int_dtype: type = np.int8,
                     float_dtype: type = np.float32):
    """df = get_drug_resp_df('./data/', 'std')

    This function loads the whole drug response file, process it and return
    as a dataframe. The processing includes:
        * remove the '-' in cell line names;
        * encode str format data sources into integer;
        * scaling the growth accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug response dataframe.
    """
    print("=" * 80)
    print("*********** get_drug_resp_df ***********")
    df_filename = 'drug_resp_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        print(f"Loading from Pickle {df_path}")
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        print("Reading from scratch...")
        logger.debug('Processing drug response dataframe ...')

        # Download the raw file if not exist
        download_files(filenames=DRUG_RESP_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))
        print("File Path: ", os.path.join(data_root, RAW_FOLDER, DRUG_RESP_FILENAME))
        t_s_load = time.time()
        # df = pd.read_csv(
        #     os.path.join(data_root, RAW_FOLDER, DRUG_RESP_FILENAME),
        #     sep='\t',
        #     header=0,
        #     index_col=None,
        #     usecols=[0, 1, 2, 4, 6, ])
        csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
            "\t")
        tb: Table = read_csv(ctx, os.path.join(data_root, RAW_FOLDER, DRUG_RESP_FILENAME),
                             csv_read_options)
        tb = tb[['SOURCE', 'DRUG_ID', 'CELLNAME', 'LOG_CONCENTRATION', 'GROWTH']]
        t_e_load = time.time()

        print(f" Data Loading Time : {t_e_load - t_s_load} s")
        #print(f" DataFrame shape {tb.shape}")
        #print("Column names: ", tb.column_names)

        # Delete '-', which could be inconsistent between seq and meta
        #df['CELLNAME'] = df['CELLNAME'].str.replace('-', '')
        t_str_replace_time = time.time()
        tb['CELLNAME'] = tb['CELLNAME'].applymap(lambda x: x.replace('-', ''))
        t_str_replace_time = time.time() - t_str_replace_time
        print(f"Str replace time {t_str_replace_time} s")
        # t_2_pdf_start = time.time()
        # df = tb.to_pandas()
        # t_2_pdf_end = time.time()
        # print(f"Pandas Conversion time: {t_2_pdf_end - t_2_pdf_start} s")
        # Encode data sources into numeric
        labels_list = tb['SOURCE'].to_numpy(zero_copy_only=False).flatten().tolist()
        encoded_list = encode_label_to_int(data_root=data_root,
                                           dict_name='data_src_dict.txt',
                                           labels=labels_list)
        tb['SOURCE'] = Table.from_list(ctx, ['SOURCE'], [encoded_list])
        # df['SOURCE'] = encode_label_to_int(data_root=data_root,
        #                                    dict_name='data_src_dict.txt',
        #                                    labels=df['SOURCE'].tolist())

        # Scaling the growth with given scaling method
        df = tb.to_pandas()
        df['GROWTH'] = scale_dataframe(df['GROWTH'], grth_scaling)
        tb = Table.from_pandas(ctx, df)
        # Convert data type into generic python types
        tb['SOURCE'] = tb['SOURCE'].astype(int)
        tb['LOG_CONCENTRATION'] = tb['LOG_CONCENTRATION'].astype(float)
        tb['GROWTH'] = tb['GROWTH'].astype(float)
        # df[['SOURCE']] = df[['SOURCE']].astype(int)
        # df[['LOG_CONCENTRATION', 'GROWTH']] = \
        #     df[['LOG_CONCENTRATION', 'GROWTH']].astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        #df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    tb['SOURCE'] = tb['SOURCE'].astype('int8')
    tb['LOG_CONCENTRATION'] = tb['LOG_CONCENTRATION'].astype('float32')
    tb['GROWTH'] = tb['GROWTH'].astype('float32')
    # df[['SOURCE']] = df[['SOURCE']].astype(int_dtype)
    # df[['LOG_CONCENTRATION', 'GROWTH']] = \
    #     df[['LOG_CONCENTRATION', 'GROWTH']].astype(float_dtype)
    print(f"Data Frame size: {tb.shape}")
    print("=" * 80)
    return tb


def get_combo_stats_df(data_root: str,
                       grth_scaling: str,
                       int_dtype: type = np.int8,
                       float_dtype: type = np.float32):
    """df = get_combo_stats_df('./data/', 'std')

    This function loads the whole drug response file, takes out every single
    drug + cell line combinations, and calculates the statistics including:
        * number of drug response records per combo;
        * average growth per combo;
        * correlation between drug log concentration and growth per combo;
    for all the combinations.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug cell combination statistics dataframe, each row
            contains the following fields: ['DRUG_ID', 'CELLNAME','NUM_REC',
            'AVG_GRTH', 'CORR'].
    """
    print("=" * 80)
    df_filename = 'combo_stats_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        logger.debug('Processing drug + cell combo statics dataframe ... '
                     'this may take up to 5 minutes.')

        # Load the whole drug response dataframe and create a combo column
        # Use generic python dtypes to minimize the error during processing
        drug_resp_tb = get_drug_resp_df(data_root=data_root,
                                        grth_scaling=grth_scaling,
                                        int_dtype=int,
                                        float_dtype=float)
        drug_resp_df = drug_resp_tb.to_pandas()
        # logger.debug('Limiting the dataframe with drugs and cell lines ... ')
        # drug_resp_df = drug_resp_df.loc[
        #     (drug_resp_df['CELLNAME'].isin(get_all_cells(data_root))) &
        #     (drug_resp_df['DRUG_ID'].isin(get_all_drugs(data_root)))]

        # Using a dict to store all combo info with a single iteration
        combo_dict = {}

        # Note that there are different ways of iterating the dataframe
        # Fastest way is to convert the dataframe into ndarray, which is
        # 2x faster than itertuples(), which is 110x faster than iterrows().
        # This part takes about 30 sec on AMD 2700X
        drug_resp_array = drug_resp_df.values

        # Each row in the drug response dataframe contains:
        # ['SOURCE', 'DRUG_ID', 'CELLNAME', 'LOG_CONCENTRATION', 'GROWTH']
        for row in drug_resp_array:

            # row[1] = drug
            # row[2] = cell
            # row[3] = concentration
            # row[4] = growth

            # The combo name is made of drug + cell line
            combo = row[1] + '+' + row[2]
            if combo not in combo_dict:
                # Each dictionary value will be a list containing:
                # [drug, cell, tuple of concentration, tuple of growth]
                combo_dict[combo] = [row[1], row[2], (), ()]

            # Concentration and growth
            combo_dict[combo][2] += (row[3], )
            combo_dict[combo][3] += (row[4], )

        # Using list of lists (table) for much faster data access
        # This part is parallelized using joblib
        def process_combo(dict_value: list):

            # Each dict value will be a list containing:
            # [drug, cell, tuple of concentration, tuple of growth]
            conc_tuple = dict_value[2]
            grth_tuple = dict_value[3]

            # This might throw warnings as var(growth) == 0 sometimes
            # Fill NaN with 0 as there is no correlation in cases like this
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    corr = stats.pearsonr(x=conc_tuple, y=grth_tuple)[0]
                except Warning:
                    corr = 0.0
                except ValueError:
                    corr = 0.0
            # corr = 0. if np.isnan(corr) else corr

            # Each row contains the following fields:
            # ['DRUG_ID', 'CELLNAME','NUM_REC', 'AVG_GRTH', 'CORR']
            return [dict_value[0], dict_value[1], len(conc_tuple),
                    np.mean(grth_tuple), corr]

        num_cores = multiprocessing.cpu_count()
        print("Num cores : ", num_cores )
        combo_stats = Parallel(n_jobs=num_cores)(
             delayed(process_combo)(v) for _, v in combo_dict.items())


        # Convert ths list of lists to dataframe
        cols = ['DRUG_ID', 'CELLNAME', 'NUM_REC', 'AVG_GRTH', 'CORR']
        df = pd.DataFrame(combo_stats, columns=cols)
        tb = Table.from_pandas(ctx, df)
        print(f">>>1 get_combo_stats_df.DRUG_ID "
              f"{tb['DRUG_ID'].to_numpy(zero_copy_only=False).tolist()[0:5]}")
        # Convert data type into generic python types
        # df[['NUM_REC']] = df[['NUM_REC']].astype(int)
        # df[['AVG_GRTH', 'CORR']] = df[['AVG_GRTH', 'CORR']].astype(float)
        tb['NUM_REC'] = tb['NUM_REC'].astype(int)
        tb['AVG_GRTH'] = tb['AVG_GRTH'].astype(float)
        tb['CORR'] = tb['CORR'].astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        #df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    # df[['NUM_REC']] = df[['NUM_REC']].astype(int_dtype)
    # df[['AVG_GRTH', 'CORR']] = df[['AVG_GRTH', 'CORR']].astype(float_dtype)
    tb['NUM_REC'] = tb['NUM_REC'].astype(int)
    tb['AVG_GRTH'] = tb['AVG_GRTH'].astype(float)
    tb['CORR'] = tb['CORR'].astype(float)
    print(f">>>2 get_combo_stats_df.DRUG_ID "
          f"{tb['DRUG_ID'].to_numpy(zero_copy_only=False).tolist()[0:5]}")
    print("=" * 80)
    return tb


def get_drug_stats_df(data_root: str,
                      grth_scaling: str,
                      int_dtype: type = np.int16,
                      float_dtype: type = np.float32):
    """df = get_drug_stats_df('./data/', 'std')

    This function loads the combination statistics file, iterates through
    all the drugs, and calculated the statistics including:
        * number of cell lines tested per drug;
        * number of drug response records per drug;
        * average of growth per drug;
        * average correlation of dose and growth per drug;
    for all the drugs.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug cell combination statistics dataframe, each row
            contains the following fields: ['DRUG_ID', 'NUM_CL', 'NUM_REC',
            'AVG_GRTH', 'AVG_CORR']
    """
    print("=" * 80)
    if int_dtype == np.int8:
        logger.warning('Integer length too smaller for drug statistics.')

    df_filename = 'drug_stats_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        logger.debug('Processing drug statics dataframe ... ')

        # Load combo (drug + cell) dataframe to construct drug statistics
        combo_stats_df = get_combo_stats_df(data_root=data_root,
                                            grth_scaling=grth_scaling,
                                            int_dtype=int,
                                            float_dtype=float)

        # Using a dict to store all drug info with a single iteration
        drug_dict = {}

        # Each row in the combo stats dataframe contains:
        # ['DRUG_ID', 'CELLNAME','NUM_REC', 'AVG_GRTH', 'CORR']
        #combo_stats_array = combo_stats_df.values
        combo_stats_array = None
        try:
            combo_stats_array = combo_stats_df.to_numpy()
        except ValueError:
            combo_stats_array = combo_stats_df.to_numpy(zero_copy_only=False)
        for row in combo_stats_array:
            drug = row[0]
            if drug not in drug_dict:
                # Each dictionary value will be a list containing:
                # [num of cell, tuple of num of records,
                #  tuple of avg grth, tuple of corr]
                drug_dict[drug] = [0, (), (), ()]

            drug_dict[drug][0] += 1
            drug_dict[drug][1] += (row[2], )
            drug_dict[drug][2] += (row[3], )
            drug_dict[drug][3] += (row[4], )

        # Using list of lists (table) for much faster data access
        # This part is parallelized using joblib
        def process_drug(drug: str, dict_value: list):

            # Each row in the drug stats dataframe contains:
            # ['DRUG_ID', 'NUM_CL', 'NUM_REC', 'AVG_GRTH', 'AVG_CORR']
            num_cl = dict_value[0]
            records_tuple = dict_value[1]

            assert num_cl == len(records_tuple)

            grth_tuple = dict_value[2]
            corr_tuple = dict_value[3]

            num_rec = np.sum(records_tuple)
            avg_grth = np.average(a=grth_tuple, weights=records_tuple)
            avg_corr = np.average(a=corr_tuple, weights=records_tuple)

            return [drug, num_cl, num_rec, avg_grth, avg_corr]

        num_cores = multiprocessing.cpu_count()
        drug_stats = Parallel(n_jobs=num_cores)(
            delayed(process_drug)(k, v) for k, v in drug_dict.items())

        # Convert ths list of lists to dataframe
        cols = ['DRUG_ID', 'NUM_CL', 'NUM_REC', 'AVG_GRTH', 'AVG_CORR']
        df = pd.DataFrame(drug_stats, columns=cols)
        tb = Table.from_pandas(ctx, df)

        # Convert data type into generic python types
        # df[['NUM_CL', 'NUM_REC']] = df[['NUM_CL', 'NUM_REC']].astype(int)
        # df[['AVG_GRTH', 'AVG_CORR']] = \
        #     df[['AVG_GRTH', 'AVG_CORR']].astype(float)
        # df.set_index('DRUG_ID', inplace=True)

        tb['NUM_CL'] = tb['NUM_CL'].astype(int)
        tb['NUM_REC'] = tb['NUM_REC'].astype(int)
        tb['AVG_GRTH'] = tb['AVG_GRTH'].astype(float)
        tb['AVG_CORR'] = tb['AVG_CORR'].astype(float)
        print(f">>>1 get_drug_stats_df.index : {tb.shape}"
              f"{tb['DRUG_ID'].to_numpy(zero_copy_only=False).flatten().tolist()[0:5]}")
        #tb.set_index('DRUG_ID', drop=True)
        # print(f">>>2 get_drug_stats_df.index : {tb.shape}"
        #       f"{tb.index.values.flatten().tolist()[0:5]}")
        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        #df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    # df[['NUM_CL', 'NUM_REC']] = df[['NUM_CL', 'NUM_REC']].astype(int_dtype)
    # df[['AVG_GRTH', 'AVG_CORR']] = \
    #     df[['AVG_GRTH', 'AVG_CORR']].astype(float_dtype)
    # print(f">>>3 get_drug_stats_df.index {tb.shape} : "
    #       f"{tb.index.values.flatten().tolist()[0:5]}")
    tb['NUM_CL'] = tb['NUM_CL'].astype('int32')
    tb['NUM_REC'] = tb['NUM_REC'].astype('int32')
    tb['AVG_GRTH'] = tb['AVG_GRTH'].astype('float32')
    tb['AVG_CORR'] = tb['AVG_CORR'].astype('float32')
    tb.set_index('DRUG_ID', drop=True)
    print(f">>>4 get_drug_stats_df.index {tb.shape}: "
          f"{tb.index.values.flatten().tolist()[0:5]}")
    print("=" * 80)
    return tb


def get_drug_anlys_df(data_root: str):
    """df = get_drug_anlys_df('./data/')

    This function will load the drug statistics dataframe and go on and
    classify all the drugs into 4 different categories:
        * high growth, high correlation
        * high growth, low correlation
        * low growth, high correlation
        * low growth, low correlation
    Using the median value of growth and correlation. The results will be
    returned as a dataframe with drug ID as index.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.

    Returns:
        pd.DataFrame: drug classes with growth and correlation, each row
            contains the following fields: ['HIGH_GROWTH', 'HIGH_CORR'],
            which are boolean features.
    """
    print("=" * 80)
    print("get_drug_anlys_df")
    df_filename = 'drug_anlys_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    print(f"df_filename: {df_filename}")
    print(f"df_path: {df_path}")

    # If the dataframe already exists, load and return ########################
    if os.path.exists(df_path):
        print("Reading from pickle")
        return pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        print("Reading from scratch")
        logger.debug('Processing drug analysis dataframe ... ')

        # Load drug statistics dataframe
        # Note that the scaling of growth has nothing to do with the analysis
        drug_stats_df = get_drug_stats_df(data_root=data_root,
                                          grth_scaling='none',
                                          int_dtype=int,
                                          float_dtype=float)

        print(f"DataFrame: {drug_stats_df.shape}")

        #drugs = drug_stats_df.index
        drugs = drug_stats_df.index.values.flatten().tolist()

        print(f">>> Drug Stats DF Index: {drugs[0:5]}")

        # avg_grth = drug_stats_df['AVG_GRTH'].values
        # avg_corr = drug_stats_df['AVG_CORR'].values
        avg_grth = None
        avg_corr = None
        try:
            avg_grth = drug_stats_df['AVG_GRTH'].to_numpy().flatten()
        except ValueError:
            avg_grth = drug_stats_df['AVG_GRTH'].to_numpy(zero_copy_only=False).flatten()

        try:
            avg_corr = drug_stats_df['AVG_CORR'].to_numpy().flatten()
        except ValueError:
            avg_corr = drug_stats_df['AVG_CORR'].to_numpy(zero_copy_only=False).flatten()

        high_grth = (avg_grth > np.median(avg_grth))
        high_corr = (avg_corr > np.median(avg_corr))

        drug_analysis_array = \
            np.array([drugs, high_grth, high_corr]).transpose()

        # The returned dataframe will have two columns of boolean values,
        # indicating four different categories.
        df = pd.DataFrame(drug_analysis_array,
                          columns=['DRUG_ID', 'HIGH_GROWTH', 'HIGH_CORR'])
        tb = Table.from_pandas(ctx, df)
        tb.set_index('DRUG_ID', drop=True)
        # df.set_index('DRUG_ID', inplace=True)

        print(f"drug_analysis_array: {drug_analysis_array.shape}")
        #print(drug_analysis_array)
        #print("response_dataframes.Index")
        #print(df.index)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        #df.to_pickle(df_path)
        print("=" * 80)
        return tb


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print('=' * 80 + '\nDrug response dataframe head:')
    t1 = time.time()
    drug_resp_tb = get_drug_resp_df(data_root='../../data/',
                           grth_scaling='none')
    drug_resp_df = drug_resp_tb.to_pandas()
    print(drug_resp_df.head())
    # Test statistic data loading functions
    print('=' * 80 + '\nDrug analysis dataframe head:')
    #print(get_drug_anlys_df(data_root='../../data/').head())

    # Plot histogram for drugs ('AVG_GRTH', 'AVG_CORR')
    # get_drug_stats_df(data_root='../../data/', grth_scaling='none'). \
    #     hist(column=['AVG_GRTH', 'AVG_CORR'], figsize=(16, 9), bins=20)
    t2 = time.time()
    # plt.suptitle('Histogram of average growth and average correlation between '
    #              'concentration and growth of all drugs')
    #plt.show()
    print(f"Time Taken: {t2-t1} s")

