""" 
    File Name:          UnoPytorch/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:
        This file implements the dataset for drug response.
"""

import logging
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.data_processing.cell_line_dataframes import get_rna_seq_df, \
    get_cl_meta_df
from utils.data_processing.drug_dataframes import get_drug_feature_df
from utils.data_processing.label_encoding import get_label_dict
from utils.data_processing.response_dataframes import get_drug_resp_df, \
    get_drug_anlys_df

logger = logging.getLogger(__name__)

# pycylon imports start

from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import time

ctx = CylonContext(config=None, distributed=False)


# pycylon imports end

class DrugRespDataset(data.Dataset):
    """Dataset class for drug response learning.

    This class implements a PyTorch Dataset class made for drug response
    learning. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of (feature, target), where feature is
    a list including drug and cell line information along with the log
    concentration, and target is the growth.

    Note that all items in feature and the target are in python float type.

    Attributes:
        training (bool): indicator of training/validation dataset.
        drugs (list): list of all the drugs in the dataset.
        cells (list): list of all the cells in the dataset.
        data_source (str): source of the data being used.
        num_records (int): number of drug response records.
        drug_feature_dim (int): dimensionality of drug feature.
        rnaseq_dim (int): dimensionality of RNA sequence.
    """

    def __init__(
            self,
            data_root: str,
            data_src: str,
            training: bool,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,

            # Pre-processing settings
            grth_scaling: str = 'none',
            dscptr_scaling: str = 'std',
            rnaseq_scaling: str = 'std',
            dscptr_nan_threshold: float = 0.0,

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'source_scale',
            drug_feature_usage: str = 'both',
            validation_ratio: float = 0.2,
            disjoint_drugs: bool = True,
            disjoint_cells: bool = True, ):
        """dataset = DrugRespDataset('./data/', 'NCI60', True)

        Construct a new drug response dataset based on the parameters
        provided. The process includes:
            * Downloading source data files;
            * Pre-processing (scaling, trimming, etc.);
            * Public attributes and other preparations.

        Args:
            data_root (str): path to data root folder.
            data_src (str): data source for drug response, must be one of
                'NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI', and 'all'.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM.
            float_dtype (type): float dtype for data storage in RAM.
            output_dtype (type): output dtype for neural network.

            grth_scaling (str): scaling method for drug response growth.
                Choose between 'none', 'std', and 'minmax'.
            dscptr_scaling (str): scaling method for drug descriptor.
                Choose between 'none', 'std', and 'minmax'.
            rnaseq_scaling (str): scaling method for RNA sequence (LINCS1K).
                Choose between 'none', 'std', and 'minmax'.
            dscptr_nan_threshold (float): NaN threshold for drug descriptor.
                If a column/feature or row/drug contains exceeding amount of
                NaN comparing to the threshold, the feature/drug will be
                dropped.

            rnaseq_feature_usage (str): RNA sequence usage. Choose between
                'combat', which is batch-effect-removed version of RNA
                sequence, or 'source_scale'.
            drug_feature_usage (str): drug feature usage. Choose between
                'fingerprint', 'descriptor', or 'both'.
            validation_ratio (float): portion of validation data out of all
                data samples. Note that this is not strictly the portion
                size. During the split, we will pick a percentage of
                drugs/cells and take the combination. The calculation will
                make sure that the expected validation size is accurate,
                but not strictly the case for a single random seed. Please
                refer to __split_drug_resp() for more details.
            disjoint_drugs (bool): indicator for disjoint drugs between
                training and validation dataset.
            disjoint_cells: indicator for disjoint cell lines between
                training and validation dataset.
        """

        # Initialization ######################################################
        print("=" * 80)
        print(f"-----DrugRespDataset[{data_src}]----")
        self.__data_root = data_root

        # Class-wise variables
        self.data_source = data_src
        self.training = training
        self.__rand_state = rand_state
        self.__output_dtype = output_dtype

        # Feature scaling
        if grth_scaling is None or grth_scaling == '':
            grth_scaling = 'none'
        grth_scaling = grth_scaling.lower()
        if dscptr_scaling is None or dscptr_scaling == '':
            dscptr_scaling = 'none'
        dscptr_scaling = dscptr_scaling
        if rnaseq_scaling is None or rnaseq_scaling == '':
            rnaseq_scaling = 'none'
        rnaseq_scaling = rnaseq_scaling

        self.__validation_ratio = validation_ratio
        self.__disjoint_drugs = disjoint_drugs
        self.__disjoint_cells = disjoint_cells

        # Load all dataframes #################################################
        self.__drug_resp_df = get_drug_resp_df(
            data_root=data_root,
            grth_scaling=grth_scaling,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        self.__drug_resp_tb = Table.from_pandas(ctx, self.__drug_resp_df)

        self.__drug_feature_df = get_drug_feature_df(
            data_root=data_root,
            drug_feature_usage=drug_feature_usage,
            dscptr_scaling=dscptr_scaling,
            dscptr_nan_thresh=dscptr_nan_threshold,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        self.__drug_feature_tb = Table.from_pandas(ctx, self.__drug_feature_df, preserve_index=True)
        self.__drug_feature_tb.set_index(self.__drug_feature_tb.column_names[-1], drop=True)

        self.__rnaseq_tb = get_rna_seq_df(
            data_root=data_root,
            rnaseq_feature_usage=rnaseq_feature_usage,
            rnaseq_scaling=rnaseq_scaling,
            float_dtype=float_dtype)

        self.__rnaseq_tb.reset_index()
        self.__rnaseq_df = self.__rnaseq_tb.to_pandas()
        self.__rnaseq_df.set_index(self.__rnaseq_df.columns[0], drop=True, inplace=True)
        self.__rnaseq_tb.set_index(self.__rnaseq_tb.column_names[0], drop=True)
        #self.__rnaseq_tb = Table.from_pandas(ctx, self.__rnaseq_df, preserve_index=True)
        #self.__rnaseq_tb.set_index(self.__rnaseq_tb.column_names[-1], drop=True)

        # Train/validation split ##############################################
        print(f"Loaded Drug Resp DF Original Amount : {self.__drug_resp_tb.shape}")
        t_split_start = time.time()
        self.__split_drug_resp()
        t_split_end = time.time()

        print(f"Split Time: {t_split_end - t_split_start} s")

        # Public attributes ###################################################
        print(f"Drugs Original Amount : {self.__drug_resp_tb.shape}")
        t_unq_start = time.time()
        tb_drugs_unique = self.__drug_resp_tb['DRUG_ID'].unique()
        tb_cells_unique = self.__drug_resp_tb['CELLNAME'].unique()
        tb_drugs_unique_list = tb_drugs_unique.to_numpy(zero_copy_only=False).tolist() #list(
        # tb_drugs_unique.to_pydict(
        # ).items())[
        # 0][1]
        tb_cells_unique_list = tb_cells_unique.to_numpy(zero_copy_only=False).tolist()#list(tb_cells_unique.to_pydict(
        # ).items())[
        # 0][1]
        t_unq_end = time.time()
        print(f"Cylon Unique Time : {t_unq_end - t_unq_start} s ")

        self.drugs = tb_drugs_unique_list  # self.__drug_resp_df['DRUG_ID'].unique().tolist()
        self.cells = tb_cells_unique_list  # self.__drug_resp_df['CELLNAME'].unique().tolist()
        self.num_records = self.__drug_resp_tb.row_count  # len(self.__drug_resp_df)
        self.drug_feature_dim = self.__drug_feature_tb.column_count  # self.__drug_feature_df.shape[1]
        self.rnaseq_dim = self.__rnaseq_tb.column_count  # self.__rnaseq_df.shape[1]
        #assert self.__rnaseq_df.shape[1] == self.__rnaseq_tb.column_count
        # #self.__rnaseq_df.shape[1]
        print(f"Drugs count : {len(tb_drugs_unique_list)}")
        print(f"Cells count : {len(tb_cells_unique_list)}")

        print(f"2*** Drug resp shapes: {self.__drug_resp_tb.shape}")
        # print(f"2*** Drug feature shapes: {self.__drug_feature_df.shape},"
        #       f" {self.__drug_feature_tb.shape}")

        # Converting dataframes to arrays and dict for rapid access ###########
        self.__drug_resp_array = None
        try:
            self.__drug_resp_array = self.__drug_resp_tb.to_numpy()
        except ValueError:
            print("Data to Numpy Non-zero-copy")
            self.__drug_resp_array = self.__drug_resp_tb.to_numpy(zero_copy_only=False)
        #self.__drug_resp_df.values

        # The following conversion will upcast dtypes
        # self.__drug_feature_dict = {idx: row.values.tolist() for idx, row in
        #                             self.__drug_feature_df.iterrows()}
        self.__drug_feature_dict = {idx: np.array(row) for idx, row in
                                    self.__drug_feature_tb.iterrows()}
        self.__rnaseq_dict = {idx: np.array(row) for idx, row in self.__rnaseq_tb.iterrows()}
        # self.__rnaseq_dict = {idx: row.values.tolist() for idx, row in
        #                       self.__rnaseq_df.iterrows()}

        # Dataframes are not needed any more
        self.__drug_resp_df = None
        self.__drug_feature_df = None
        self.__rnaseq_df = None

        self.__drug_resp_tb = None
        self.__drug_feature_tb = None
        self.__rnaseq_tb = None

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(('Training' if self.training else 'Validation')
                  + ' Drug Response Dataset Summary (Data Source: %6s):'
                  % self.data_source)
            print('\t%i Drug Response Records .' % len(self.__drug_resp_array))
            print('\t%i Unique Drugs (feature dim: %4i).'
                  % (len(self.drugs), self.drug_feature_dim))
            print('\t%i Unique Cell Lines (feature dim: %4i).'
                  % (len(self.cells), self.rnaseq_dim))
            print('=' * 80)
        print("=" * 80)

    def __len__(self):
        return self.num_records

    def __getitem__(self, index):
        """rnaseq, drug_feature, concentration, growth = dataset[0]

        This function fetches a single sample of drug response data along
        with the corresponding drug features and RNA sequence.

        Note that all the returned values are in ndarray format with the
        type specified during dataset initialization.

        Args:
            index (int): index for drug response data.

        Returns:
            tuple: a tuple of np.ndarray, with RNA sequence data,
                drug features, concentration, and growth.
        """

        # Note that this chunk of code does not work with pytorch 4.1 for
        # multiprocessing reasons. Chances are that during the run, one of
        # the workers might hang and prevents the training from moving on

        # Note that even with locks for multiprocessing lib, the code will
        # get stuck at some point if all CPU cores are used.

        drug_resp = self.__drug_resp_array[index]
        drug_feature = self.__drug_feature_dict[drug_resp[1]]
        rnaseq = self.__rnaseq_dict[drug_resp[2]]

        drug_feature = drug_feature.astype(self.__output_dtype)
        rnaseq = rnaseq.astype(self.__output_dtype)
        concentration = np.array([drug_resp[3]], dtype=self.__output_dtype)
        growth = np.array([drug_resp[4]], dtype=self.__output_dtype)

        return rnaseq, drug_feature, concentration, growth

    def __trim_dataframes(self):
        """self.__trim_dataframes(trim_data_source=True)

        This function trims three dataframes to make sure that drug response
        dataframe, RNA sequence dataframe, and drug feature dataframe are
        sharing the same list of cell lines and drugs.

        Returns:
            None
        """

        # Encode the data source and take the data from target source only
        # Note that source could be 'NCI60', 'GDSC', etc. and 'all'
        print("=" * 80)
        print("__trim_dataframes")
        print("")
        t_trim_time = time.time()
        if self.data_source.lower() != 'all':
            print("self.data_source.lower() != 'all'")
            logger.debug('Specifying data source %s ... ' % self.data_source)

            data_src_dict = get_label_dict(data_root=self.__data_root,
                                           dict_name='data_src_dict.txt')
            encoded_data_src = data_src_dict[self.data_source]

            print(f"=====> encode_data_src {type(encoded_data_src)}")
            t_filter_by_val = time.time()
            reduction_trim_tb = self.__drug_resp_tb['SOURCE'] == encoded_data_src
            reduction_trim_tb_list = reduction_trim_tb.to_numpy(
                zero_copy_only=False).flatten().tolist()
            #reduction_trim_tb_list = list(reduction_trim_tb.to_pydict().items())[0][1]
            t_filter_by_val = time.time() - t_filter_by_val
            print(f"Time taken for filter data table {t_filter_by_val} s")

            # reduction_trim_series = self.__drug_resp_df['SOURCE'] == encoded_data_src
            # reduction_trim_list = reduction_trim_series.tolist()

            # assert reduction_trim_tb_list == reduction_trim_list

            print(
                f"=====> Reduction Trim Df : {type(reduction_trim_tb)}, size: {len(reduction_trim_tb_list)}")

            # Reduce/trim the drug response dataframe
            print(f"1.$$$ self.__drug_resp_df.shape : {self.__drug_resp_df.shape} "
                  f"{self.__drug_resp_tb.shape}")
            t2 = time.time()
            self.__drug_resp_df = self.__drug_resp_df.loc[reduction_trim_tb_list]
            t3 = time.time()
            # self.__drug_resp_tb = self.__drug_resp_tb.loc[reduction_trim_tb_list]
            # t4 = time.time()
            # print(f"2.$$$ self.__drug_resp_df.shape : {self.__drug_resp_df.shape} "
            #       f"{self.__drug_resp_tb.shape}, {t3 - t2}, {t4 - t3}")
            self.__drug_resp_tb = Table.from_pandas(ctx, self.__drug_resp_df)
            t4 = time.time()
            print(f"3.$$$ self.__drug_resp_df.shape : {self.__drug_resp_df.shape} "
                  f"{self.__drug_resp_tb.shape}, {t3 - t2} {t4-t3}")

        # Make sure that all three dataframes share the same drugs/cells
        logger.debug('Trimming dataframes on common cell lines and drugs ... ')
        print(">>>> Shape of Overlapping cell name and index values")
        # print(f"Overlapping Shapes {len(self.__rnaseq_df.index.values)}, "
        #       f"{self.__drug_resp_df['CELLNAME'].unique().shape}")
        #print(f"RNASEQ Index values : {self.__rnaseq_df.index.values.shape}")
        # print(f"Drug response unique: {self.__drug_resp_df['CELLNAME'].unique()}")

        # drug_res_cell_unique = self.__drug_resp_df['CELLNAME'].unique()  # this is numpy ndarray
        # drug_res_drug_unique = self.__drug_resp_df['DRUG_ID'].unique()
        t_unique_op_1 = time.time()
        drug_res_cell_unique_tb = self.__drug_resp_tb['CELLNAME'].unique()
        drug_res_cell_unique_np = drug_res_cell_unique_tb.to_numpy(
            zero_copy_only=False).flatten()#np.array(list(
        # drug_res_cell_unique_tb.to_pydict().items())[0][1])

        drug_res_drug_unique_tb = self.__drug_resp_tb['DRUG_ID'].unique()
        drug_res_drug_unique_np = drug_res_drug_unique_tb.to_numpy(
            zero_copy_only=False).flatten()#np.array(list(
        # drug_res_drug_unique_tb.to_pydict().items())[0][1])
        t_unique_op_1 = time.time() - t_unique_op_1
        print(f"Trim Unique Op 1 Time: {t_unique_op_1} s")

        # print(f"1. Unique Op Shapes {drug_res_cell_unique.shape}, {drug_res_cell_unique_np.shape}")
        # print(f"2. Unique Op Shapes {drug_res_drug_unique.shape}, {drug_res_drug_unique_np.shape}")

        # assert drug_res_cell_unique.tolist() == drug_res_cell_unique_np.tolist()
        # assert drug_res_drug_unique.tolist() == drug_res_drug_unique_np.tolist()

        rnaseq_index_values = self.__rnaseq_tb.index.values # self.__rnaseq_df.index.values
        # this is numpy ndarray
        drug_feature_index_values = self.__drug_feature_tb.index.values
        #self.__drug_feature_df.index.values

        #assert drug_feature_index_values.tolist() == self.__drug_feature_tb.index.index_values
        #assert self.__rnaseq_df.index.values.tolist() == self.__rnaseq_tb.index.index_values
        cell_set_formation_time = time.time()
        cell_set = list(set(drug_res_cell_unique_np) & set(rnaseq_index_values))
        drug_set = list(set(drug_res_drug_unique_np) & set(drug_feature_index_values))
        cell_set_formation_time = time.time() - cell_set_formation_time
        print(f"Cell Set Formation Time: {cell_set_formation_time} s")
        print(f"a: self.__drug_resp_df.shape : {self.__drug_resp_df.shape}, "
              f"{self.__drug_resp_tb.shape}")
        print(f">> cell_set {len(cell_set)}, {type(cell_set)},  {type(cell_set[0])}")
        print(f">> drug_set {len(drug_set)}, {type(drug_set)}, {type(drug_set[0])}")
        #print(f">> drug_set_index {len(self.__drug_feature_df.index.values)}")

        # drug_resp_df_cell_isin = self.__drug_resp_df['CELLNAME'].isin(cell_set)
        # drug_resp_df_drugid_isin = self.__drug_resp_df['DRUG_ID'].isin(drug_set)
        t_isin_time_op_1 = time.time()
        drug_resp_tb_cell_isin = self.__drug_resp_tb['CELLNAME'].isin(cell_set)
        drug_resp_tb_drugid_isin = self.__drug_resp_tb['DRUG_ID'].isin(drug_set)
        t_isin_time_op_1 = time.time() - t_isin_time_op_1
        print(f"Is in op 1 time: {t_isin_time_op_1} s")
        t_isin_filter_cr = time.time()
        # drug_resp_tb_cell_isin_np = np.array(list(drug_resp_tb_cell_isin.to_pydict().items())[0][1])
        # drug_resp_tb_drugid_isin_np = np.array(
        #     list(drug_resp_tb_drugid_isin.to_pydict().items())[0][1])
        drug_resp_tb_cell_isin_np = drug_resp_tb_cell_isin.to_numpy(zero_copy_only=False)
        drug_resp_tb_drugid_isin_np = drug_resp_tb_drugid_isin.to_numpy(zero_copy_only=False)
        # drug_resp_df_loc_filters = (drug_resp_df_cell_isin) & (drug_resp_df_drugid_isin)

        drug_resp_tb_loc_filters = (drug_resp_tb_cell_isin_np) & (drug_resp_tb_drugid_isin_np)
        t_isin_filter_cr = time.time() - t_isin_filter_cr
        print(f"Isin Filter creation time : {t_isin_filter_cr} s ["
              f"{drug_resp_tb_cell_isin_np.shape}, {drug_resp_tb_drugid_isin_np.shape}]")


        # assert drug_resp_df_loc_filters.tolist() == drug_resp_tb_loc_filters.tolist()

        print(f">>> drug_resp_df_loc_filters : {type(drug_resp_tb_loc_filters)}")
        t_tb_loc_filter_time = time.time()
        self.__drug_resp_df = self.__drug_resp_df.loc[drug_resp_tb_loc_filters]
        t_tb_loc_filter_time = time.time() - t_tb_loc_filter_time
        t_from_pandas_op_1_time = time.time()
        self.__drug_resp_tb = Table.from_pandas(ctx, self.__drug_resp_df)
        print(f"From Pandas Op 1 : {time.time() - t_from_pandas_op_1_time} s")
        print(f"Tb.loc Filter Op 1 : {t_tb_loc_filter_time} s")
        print(f"b: self.__drug_resp_df.shape : {self.__drug_resp_df.shape}, "
              f"{self.__drug_resp_tb.shape}")


        #print(f"1. __rnaseq_df {self.__rnaseq_df.shape} {self.__rnaseq_tb.shape}")
        #print(f"1. __drug_feature_df {self.__drug_feature_df.shape}
        # {self.__drug_feature_tb.shape}")
        #rnaseq_df_index_isin = self.__rnaseq_df.index.isin(cell_set)
        #drug_feature_df_index_isin = self.__drug_feature_df.index.isin(drug_set)
        t_indexing_isin_filter = time.time()
        rnaseq_tb_index_isin = self.__rnaseq_tb.index.isin(cell_set)
        drug_feature_tb_index_isin = self.__drug_feature_tb.index.isin(drug_set)
        t_indexing_isin_filter = time.time() - t_indexing_isin_filter
        print(f"Indexing Isin Op Time : {t_indexing_isin_filter} s")
        #assert rnaseq_df_index_isin.tolist() == rnaseq_tb_index_isin.tolist()
        #assert drug_feature_df_index_isin.tolist() == drug_feature_tb_index_isin.tolist()

        #self.__rnaseq_df = self.__rnaseq_df[rnaseq_df_index_isin]
        #self.__drug_feature_df = self.__drug_feature_df[drug_feature_df_index_isin]
        t_index_based_tb_update = time.time()
        rnaseq_tb_filter = Table.from_list(ctx, ['filter'], [rnaseq_tb_index_isin.tolist()])
        drug_feature_tb_filter = Table.from_list(ctx, ['filter'],
                                                 [drug_feature_tb_index_isin.tolist()])
        rnaseq_new_index = self.__rnaseq_tb.index.values[rnaseq_tb_index_isin].tolist()
        self.__rnaseq_tb = self.__rnaseq_tb[rnaseq_tb_filter]
        self.__rnaseq_tb.set_index(rnaseq_new_index)
        drug_feature_new_index = self.__drug_feature_tb.index.values[
            drug_feature_tb_index_isin].tolist()
        self.__drug_feature_tb = self.__drug_feature_tb[drug_feature_tb_filter]
        self.__drug_feature_tb.set_index(drug_feature_new_index)
        t_index_based_tb_update = time.time() - t_index_based_tb_update
        print(f"Index based Table filter time : {t_index_based_tb_update} s")
        #print(f"2. __rnaseq_df {self.__rnaseq_df.shape} {self.__rnaseq_tb.shape}")
        #print(f"2. __drug_feature_df {self.__drug_feature_df.shape}
        # {self.__drug_feature_tb.shape}")

        print(f"c: self.__drug_resp_df.shape : {self.__drug_resp_df.shape}, "
              f"{self.__drug_resp_tb.shape}")
        logger.debug('There are %i drugs and %i cell lines, with %i response '
                     'records after trimming.'
                     % (len(drug_set), len(cell_set),
                        len(self.__drug_resp_df)))
        t_trim_time = time.time() - t_trim_time
        print(f"Time taken for trim drug_resp_dataset {t_trim_time} s")
        print("=" * 80)
        return

    def __split_drug_resp(self):
        """self.__split_drug_resp()

        This function split training and validation drug response data based
        on the splitting specifications (disjoint drugs and/or disjoint cells).

        Upon the call, the function summarize all the drugs and cells. If
        disjoint (drugs/cells) is set to True, then it will split the list
        (of drugs/cells) into training/validation (drugs/cells).

        Otherwise, if disjoint (drugs/cells) is set to False, we make sure
        that the training/validation set contains the same (drugs/cells).

        Then it trims all three dataframes to make sure all the data in RAM is
        relevant for training/validation

        Note that the validation size is not guaranteed during splitting.
        What the function really splits by the ratio is the list of
        drugs/cell lines. Also, if both drugs and cell lines are marked
        disjoint, the function will split drug and cell lists with ratio of
        (validation_size ** 0.7).

        Warnings will be raise if the validation ratio is off too much.

        Returns:
            None
        """

        # Trim dataframes based on data source and common drugs/cells
        # Now drug response dataframe contains training + validation
        # data samples from the same data source, like 'NCI60'
        print("=" * 80)
        print("__split_drug_resp")
        t_trim_start = time.time()
        self.__trim_dataframes()
        t_trim_end = time.time()
        print(f"Time Taken To Trim Operation : {t_trim_end - t_trim_start} s")
        t_split_time = time.time()
        # Get lists of all drugs & cells corresponding from data source
        print(f"cell_list : {self.__drug_resp_df.shape}, {self.__drug_resp_tb.shape}")
        t_unique_lst_start = time.time()
        # cell_list = self.__drug_resp_df['CELLNAME'].unique().tolist()
        # drug_list = self.__drug_resp_df['DRUG_ID'].unique().tolist()

        cell_unq_tb = self.__drug_resp_tb['CELLNAME'].unique()
        cell_list = list(cell_unq_tb.to_pydict().items())[0][1]
        drug_unq_tb = self.__drug_resp_tb['DRUG_ID'].unique()
        drug_list = list(drug_unq_tb.to_pydict().items())[0][1]
        t_end_lst_end = time.time()

        # assert cell_list == cell_tb_list
        # assert drug_list == drug_tb_list

        print(f">>> Time Taken To Unique List : {t_end_lst_end - t_unique_lst_start} s")

        print(f"cell_list : {len(cell_list)}")
        print(f"drug_list : {len(drug_list)}")

        # Create an array to store all drugs' analysis results
        t_load_drug_start = time.time()
        drug_analys_df = get_drug_anlys_df(self.__data_root)
        t_load_drug_end = time.time()
        cl_meta_tb = get_cl_meta_df(self.__data_root)
        cl_meta_tb.reset_index()
        cl_meta_df = cl_meta_tb.to_pandas()
        cl_meta_df.set_index(cl_meta_df.columns[0], drop=True, inplace=True)
        cl_meta_tb.set_index(cl_meta_tb.column_names[0], drop=True)
        t_load_cl_meta_end = time.time()

        drug_analys_tb = Table.from_pandas(ctx, drug_analys_df, preserve_index=True)
        drug_analys_tb.set_index(drug_analys_tb.column_names[-1], drop=True)

        #cl_meta_tb = Table.from_pandas(ctx, cl_meta_df, preserve_index=True)
        #cl_meta_tb.set_index(cl_meta_tb.column_names[-1], drop=True)

        assert drug_analys_tb.index.values.tolist() == drug_analys_df.index.values.tolist()
        assert cl_meta_tb.index.values.tolist() == cl_meta_df.index.values.tolist()

        print(f">>> Shape of Drug Analys Data : {drug_analys_df.shape} {drug_analys_tb.shape}")
        print(f">>> Shape of Cell Meta Data : {cl_meta_df.shape} {cl_meta_tb.shape}")

        #print(f">>> $$$ Drug Analys Results: {drug_analys_df.index.values}")
        #print(f">>> $$$ Cell Meta Results: {cl_meta_df.index.values}")
        print(f">>> Time taken to load drug meta data: {t_load_drug_end - t_load_drug_start} s")
        print(f">>> Time taken to load cell meta data: {t_load_cl_meta_end - t_load_drug_end} s")
        t_itr_start = time.time()

        drug_anlys_tb_dict = {idx: row for idx, row in drug_analys_tb.iterrows()}
        drug_anlys_tb_array = np.array([drug_anlys_tb_dict[d] for d in drug_list])

        # drug_anlys_dict = {idx: row.values for idx, row in drug_analys_df.iterrows()}
        # drug_anlys_array = np.array([drug_anlys_dict[d] for d in drug_list])

        # assert drug_analys_tb_array.tolist() == drug_anlys_array.tolist()

        # Create a list to store all cell lines types
        cell_type_tb_dict = {idx: row for idx, row in cl_meta_tb[['type']].iterrows()}
        cell_type_tb_list = [cell_type_tb_dict[c] for c in cell_list]

        # cell_type_dict = {idx: row.values for idx, row in cl_meta_df[['type']].iterrows()}
        # cell_type_list = [cell_type_dict[c] for c in cell_list]

        # print(f"cell_type dict len : "
        #       f"{len(list(cell_type_tb_dict.keys()))}, {len(list(cell_type_dict.keys()))}")
        # assert list(cell_type_tb_dict.keys()) == list(cell_type_dict.keys())
        # assert list(cell_type_tb_dict.values()) == list(cell_type_dict.values())

        # assert cell_type_tb_list == cell_type_list

        t_itr_end = time.time()

        print(f">>> Time Taken To Itr Op : {t_itr_end - t_itr_start} s")

        # Change validation size when both features are disjoint in splitting
        # Note that theoretically should use validation_ratio ** 0.5,
        # but 0.7 simply works better in most cases.
        if self.__disjoint_cells and self.__disjoint_drugs:
            adjusted_val_ratio = self.__validation_ratio ** 0.7
        else:
            adjusted_val_ratio = self.__validation_ratio

        split_kwargs = {
            'test_size': adjusted_val_ratio,
            'random_state': self.__rand_state,
            'shuffle': True, }

        # Try to split the cells stratified on type list
        try:
            t1 = time.time()
            training_cell_list, validation_cell_list = \
                train_test_split(cell_list, **split_kwargs,
                                 stratify=cell_type_tb_list)
            t2 = time.time()
            print(f">>> 1. Sub Split Timing {t2 - t1} s")
        except ValueError:
            logger.warning('Failed to split %s cells in stratified '
                           'way. Splitting randomly ...' % self.data_source)
            t1 = time.time()
            training_cell_list, validation_cell_list = \
                train_test_split(cell_list, **split_kwargs)
            t2 = time.time()
            print(f">>> 2. Sub Split Timing {t2 - t1} s")

        # Try to split the drugs stratified on the drug analysis results
        try:
            t1 = time.time()
            training_drug_list, validation_drug_list = \
                train_test_split(drug_list, **split_kwargs,
                                 stratify=drug_anlys_tb_array)
            t2 = time.time()
            print(f">>> 3. Sub Split Timing {t2 - t1} s")
        except ValueError:
            logger.warning('Failed to split %s drugs stratified on growth '
                           'and correlation. Splitting solely on avg growth'
                           ' ...' % self.data_source)

            try:
                t1 = time.time()
                training_drug_list, validation_drug_list = \
                    train_test_split(drug_list, **split_kwargs,
                                     stratify=drug_anlys_tb_array[:, 0])
                t2 = time.time()
                print(f">>> 4. Sub Split Timing {t2 - t1} s")
            except ValueError:
                logger.warning('Failed to split %s drugs on avg growth. '
                               'Splitting solely on avg correlation ...'
                               % self.data_source)

                try:
                    t1 = time.time()
                    training_drug_list, validation_drug_list = \
                        train_test_split(drug_list, **split_kwargs,
                                         stratify=drug_anlys_tb_array[:, 1])
                    t2 = time.time()
                    print(f">>> 5. Sub Split Timing {t2 - t1} s")
                except ValueError:
                    logger.warning('Failed to split %s drugs on avg '
                                   'correlation. Splitting randomly ...'
                                   % self.data_source)
                    t1 = time.time()
                    training_drug_list, validation_drug_list = \
                        train_test_split(drug_list, **split_kwargs)
                    t2 = time.time()
                    print(f">>> 6. Sub Split Timing {t2 - t1} s")

        # Split data based on disjoint cell/drug strategy
        print(f">>> $#* Before Split Stage: {self.__drug_resp_df.shape},"
              f" {self.__drug_resp_tb.shape}")
        if self.__disjoint_cells and self.__disjoint_drugs:
            print(">>> Filter Case 1")
            t1 = time.time()
            # train_cell_df_filter = self.__drug_resp_df['CELLNAME'].isin(training_cell_list)
            # train_drug_df_filter = self.__drug_resp_df['DRUG_ID'].isin(training_drug_list)
            # train_res_filter = train_cell_df_filter & train_drug_df_filter

            train_cell_tb_filter = self.__drug_resp_tb['CELLNAME'].isin(training_cell_list)
            train_drug_tb_filter = self.__drug_resp_tb['DRUG_ID'].isin(training_drug_list)
            train_res_tb_filter = train_cell_tb_filter & train_drug_tb_filter

            # assert train_res_filter.values.tolist() == train_res_tb_filter.to_pandas(
            # ).values.flatten().tolist()

            train_res_tb_filter = list(train_res_tb_filter.to_pydict().items())[0][1]

            training_drug_resp_df = self.__drug_resp_df.loc[train_res_tb_filter]

            # validation_cell_df_filter = self.__drug_resp_df['CELLNAME'].isin(validation_cell_list)
            # validation_drug_df_filter = self.__drug_resp_df['DRUG_ID'].isin(training_drug_list)
            # validation_res_filter = validation_cell_df_filter & validation_drug_df_filter

            validation_cell_tb_filter = self.__drug_resp_tb['CELLNAME'].isin(validation_cell_list)
            validation_drug_tb_filter = self.__drug_resp_tb['DRUG_ID'].isin(training_drug_list)
            validation_res_tb_filter = validation_cell_tb_filter & validation_drug_tb_filter

            # assert validation_res_filter.values.tolist() == validation_res_tb_filter.to_pandas(
            # ).values.flatten().tolist()

            # print(f"\t Filter Sizes {train_res_filter.shape} {validation_res_filter.shape}")

            validation_res_tb_filter = list(validation_res_tb_filter.to_pydict().items())[0][1]

            validation_drug_resp_df = self.__drug_resp_df.loc[validation_res_tb_filter]

            t2 = time.time()
            print(f">>> 1. Sub Loc[isin] Op {t2 - t1} s")

        elif self.__disjoint_cells and (not self.__disjoint_drugs):
            print(">>> Filter Case 2")
            t1 = time.time()
            # train_filter = self.__drug_resp_df['CELLNAME'].isin(training_cell_list)
            train_tb_filter = self.__drug_resp_tb['CELLNAME'].isin(training_cell_list)

            # assert train_tb_filter.to_pandas().values.flatten().tolist() == \
            #        train_filter.values.tolist()

            train_tb_filter = list(train_tb_filter.to_pydict().items())[0][1]

            training_drug_resp_df = self.__drug_resp_df.loc[train_tb_filter]

            # validation_filter = self.__drug_resp_df['CELLNAME'].isin(validation_cell_list)
            validation_tb_filter = self.__drug_resp_tb['CELLNAME'].isin(validation_cell_list)

            # assert validation_tb_filter.to_pandas().values.flatten().tolist() == \
            #        validation_filter.values.tolist()

            validation_tb_filter = list(validation_tb_filter.to_pydict().items())[0][1]

            validation_drug_resp_df = self.__drug_resp_df.loc[validation_tb_filter]
            t2 = time.time()
            print(f">>> 2. Sub Loc[isin] Op {t2 - t1} s")

        elif (not self.__disjoint_cells) and self.__disjoint_drugs:
            print(">>> Filter Case 3")
            t1 = time.time()
            # train_filter = self.__drug_resp_df['DRUG_ID'].isin(training_drug_list)
            train_tb_filter = self.__drug_resp_tb['DRUG_ID'].isin(training_drug_list)

            # assert train_tb_filter.to_pandas().values.flatten().tolist() ==
            # train_filter.values.tolist()

            train_tb_filter = list(train_tb_filter.to_pydict().items())[0][1]

            training_drug_resp_df = self.__drug_resp_df.loc[train_tb_filter]

            # validation_filter = self.__drug_resp_df['DRUG_ID'].isin(validation_drug_list)
            validation_tb_filter = self.__drug_resp_tb['DRUG_ID'].isin(validation_drug_list)

            # assert validation_tb_filter.to_pandas().values.flatten().tolist() == \
            #        validation_filter.values.tolist()

            validation_tb_filter = list(validation_tb_filter.to_pydict().items())[0][1]

            validation_drug_resp_df = self.__drug_resp_df.loc[validation_tb_filter]
            t2 = time.time()
            print(f">>> 3. Sub Loc[isin] Op {t2 - t1} s")

        else:
            print(">>> Filter Case 4")
            print(">>> ELSE SPLIT")
            t1 = time.time()
            training_drug_resp_df, validation_drug_resp_df = \
                train_test_split(self.__drug_resp_df,
                                 test_size=self.__validation_ratio,
                                 random_state=self.__rand_state,
                                 shuffle=False)
            t2 = time.time()
            print(f">>> 4. Sub Loc[isin] Op {t2 - t1} s")
        # Make sure that if not disjoint, the training/validation set should
        #  share the same drugs/cells
        if not self.__disjoint_cells:
            # Make sure that cell lines are common
            print(">>> Filter Case 5")
            t1 = time.time()
            train_cell_unique_df_np = training_drug_resp_df['CELLNAME'].unique()
            validation_cell_unique_df_np = validation_drug_resp_df['CELLNAME'].unique()

            common_cells = list(set(train_cell_unique_df_np) & set(validation_cell_unique_df_np))

            train_filter = training_drug_resp_df['CELLNAME'].isin(common_cells)
            training_drug_resp_df = training_drug_resp_df.loc[train_filter]

            validation_filter = validation_drug_resp_df['CELLNAME'].isin(common_cells)
            validation_drug_resp_df = validation_drug_resp_df.loc[validation_filter]
            t2 = time.time()
            print(f">>> 5. Sub Loc[isin] Op {t2 - t1} s")

        if not self.__disjoint_drugs:
            # Make sure that drugs are common
            print(">>> Filter Case 6")
            t1 = time.time()
            train_df_unq_val_np = training_drug_resp_df['DRUG_ID'].unique()
            validation_df_unq_val_np = validation_drug_resp_df['DRUG_ID'].unique()
            common_drugs = set(train_df_unq_val_np) & set(validation_df_unq_val_np)

            train_filter = training_drug_resp_df['DRUG_ID'].isin(common_drugs)
            training_drug_resp_df = training_drug_resp_df.loc[train_filter]
            validation_filter = validation_drug_resp_df['DRUG_ID'].isin(common_drugs)
            validation_drug_resp_df = validation_drug_resp_df.loc[validation_filter]
            t2 = time.time()
            print(f">>> 6. Sub Loc[isin] Op {t2 - t1} s")

        # Check if the validation ratio is in a reasonable range
        validation_ratio = len(validation_drug_resp_df) \
                           / (len(training_drug_resp_df) + len(validation_drug_resp_df))
        if (validation_ratio < self.__validation_ratio * 0.8) \
                or (validation_ratio > self.__validation_ratio * 1.2):
            logger.warning('Bad validation ratio: %.3f' %
                           validation_ratio)

        # Keep only training_drug_resp_df or validation_drug_resp_df
        self.__drug_resp_df = training_drug_resp_df if self.training \
            else validation_drug_resp_df
        self.__drug_resp_tb = Table.from_pandas(ctx, self.__drug_resp_df, preserve_index=True)
        self.__drug_resp_tb.set_index(self.__drug_resp_tb.column_names[-1], drop=True)

        #print(f"============> Drug Resp DF Index Values{self.__drug_resp_df.index.values}")

        print(f">>> $#* After Split Stage: {self.__drug_resp_df.shape},"
              f" {self.__drug_resp_tb.shape}")
        t_split_time = time.time() - t_split_time
        print(f"Time taken for Split Operation {t_split_time} s")
        print("=" * 80)


# Test segment for drug response dataset
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    t1 = time.time()
    for src in ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'][:1]:
        kwarg = {
            'data_root': '../../data/',
            'summary': False,
            'rand_state': 0, }

        trn_set = DrugRespDataset(
            data_src=src,
            training=True,
            **kwarg)

        val_set = DrugRespDataset(
            data_src=src,
            training=False,
            **kwarg)

        print('There are %i drugs and %i cell lines in %s.'
              % ((len(trn_set.drugs) + len(val_set.drugs)),
                 (len(trn_set.cells) + len(val_set.cells)), src))
    t2 = time.time()
    print(f"Total Time taken: {t2-t1} s")