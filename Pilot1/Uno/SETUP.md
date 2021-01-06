# cylon_applications

## UNO

### Set Up Environment

```bash
module load anaconda
conda create --name ENVSCIAPPS python=3.7
conda activate ENVSCIAPPS
conda install -c anaconda hdf5 theano pandas scikit-learn matplotlib
conda install -c conda-forge keras=2
conda install numba
pip3 install astropy
pip3 install patsy
pip3 install statsmodels
conda install -c conda-forge tensorflow==1.14.0
pip3 install tables
pip3 install requests

git clone git@github.com:vibhatha/Benchmarks.git
cd Benchmarks/
cd Pilot1/Uno

git branch
git checkout loocv
```

### Uno Sample Run Logs

#### Command

```bash
python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True
```

#### Log

```bash
Importing candle utils for keras
Configuration file:  /home/vibhatha/sandbox/UNO/Benchmarks/Pilot1/Uno/uno_default_model.txt
{'activation': 'relu',
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'cell_features': ['rnaseq'],
 'cell_types': None,
 'cv': 1,
 'dense': [1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'drop': 0,
 'drug_features': ['descriptors', 'fingerprints'],
 'epochs': 10,
 'feature_subsample': 0,
 'learning_rate': None,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'no_gen': False,
 'optimizer': 'adam',
 'reduce_lr': False,
 'residual': False,
 'rng_seed': 2018,
 'save_path': 'save/uno',
 'scaling': 'std',
 'solr_root': '',
 'test_sources': ['train'],
 'timeout': 3600,
 'train_sources': ['GDSC', 'CTRP', 'ALMANAC'],
 'validation_split': 0.2,
 'verbose': False,
 'warmup_lr': False}
Params:
{'activation': 'relu',
 'agg_dose': None,
 'base_lr': None,
 'batch_normalization': False,
 'batch_size': 32,
 'by_cell': None,
 'by_drug': None,
 'cache': 'cache/all',
 'cell_feature_subset_path': '',
 'cell_features': ['rnaseq'],
 'cell_subset_path': '',
 'cell_types': None,
 'cp': False,
 'cv': 1,
 'datatype': <class 'numpy.float32'>,
 'dense': [1000, 1000, 1000],
 'dense_feature_layers': [1000, 1000, 1000],
 'drop': 0,
 'drug_feature_subset_path': '',
 'drug_features': ['descriptors', 'fingerprints'],
 'drug_median_response_max': 1,
 'drug_median_response_min': -1,
 'drug_subset_path': '',
 'epochs': 10,
 'es': False,
 'experiment_id': 'EXP000',
 'export_csv': None,
 'export_data': None,
 'feature_subsample': 0,
 'feature_subset_path': '',
 'gpus': [],
 'growth_bins': 0,
 'initial_weights': None,
 'learning_rate': None,
 'logfile': None,
 'loss': 'mse',
 'max_val_loss': 1.0,
 'no_feature_source': True,
 'no_gen': False,
 'no_response_source': True,
 'optimizer': 'adam',
 'output_dir': '/home/vibhatha/sandbox/UNO/Benchmarks/Pilot1/Uno/Output/EXP000/RUN000',
 'partition_by': None,
 'preprocess_rnaseq': 'source_scale',
 'reduce_lr': False,
 'residual': False,
 'rng_seed': 2018,
 'run_id': 'RUN000',
 'save_path': 'save/uno',
 'save_weights': None,
 'scaling': 'std',
 'shuffle': False,
 'single': False,
 'solr_root': '',
 'tb': False,
 'tb_prefix': 'tb',
 'test_sources': ['train'],
 'timeout': 3600,
 'train_bool': True,
 'train_sources': ['all'],
 'use_exported_data': None,
 'use_filtered_genes': False,
 'use_landmark_genes': True,
 'validation_split': 0.2,
 'verbose': None,
 'warmup_lr': False}
WARNING:tensorflow:From uno_baseline_keras2.py:48: The name tf.set_random_seed is deprecated. Please use tf.compat.v1.set_random_seed instead.

Params: {'train_sources': ['all'], 'test_sources': ['train'], 'cell_types': None, 'cell_features': ['rnaseq'], 'drug_features': ['descriptors', 'fingerprints'], 'dense': [1000, 1000, 1000], 'dense_feature_layers': [1000, 1000, 1000], 'activation': 'relu', 'loss': 'mse', 'optimizer': 'adam', 'scaling': 'std', 'drop': 0, 'epochs': 10, 'batch_size': 32, 'validation_split': 0.2, 'cv': 1, 'max_val_loss': 1.0, 'learning_rate': None, 'base_lr': None, 'residual': False, 'reduce_lr': False, 'warmup_lr': False, 'batch_normalization': False, 'feature_subsample': 0, 'rng_seed': 2018, 'save_path': 'save/uno', 'no_gen': False, 'verbose': None, 'solr_root': '', 'timeout': 3600, 'logfile': None, 'train_bool': True, 'experiment_id': 'EXP000', 'run_id': 'RUN000', 'shuffle': False, 'gpus': [], 'agg_dose': None, 'by_cell': None, 'by_drug': None, 'cell_subset_path': '', 'drug_subset_path': '', 'drug_median_response_min': -1, 'drug_median_response_max': 1, 'no_feature_source': True, 'no_response_source': True, 'use_landmark_genes': True, 'use_filtered_genes': False, 'feature_subset_path': '', 'cell_feature_subset_path': '', 'drug_feature_subset_path': '', 'preprocess_rnaseq': 'source_scale', 'es': False, 'cp': False, 'tb': False, 'tb_prefix': 'tb', 'partition_by': None, 'cache': 'cache/all', 'single': False, 'export_csv': None, 'export_data': None, 'use_exported_data': None, 'growth_bins': 0, 'initial_weights': None, 'save_weights': None, 'datatype': <class 'numpy.float32'>, 'output_dir': '/home/vibhatha/sandbox/UNO/Benchmarks/Pilot1/Uno/Output/EXP000/RUN000'}
Cache parameter file does not exist: cache/all.params.json
Loading data from scratch ...
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/rescaled_combined_single_drug_growth
1456185344/1456184548 [==============================] - 391s

====================================================
load_single_dose_response : Time = 416.46823620796204 s
====================================================
Loaded 27769716 single drug dose response measurements
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/ComboDrugGrowth_Nov2017.csv
611483648/611508384 [============================>.] - ETA: 0s
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/NCI60_CELLNAME_to_Combo.txt
8192/2819 [=======================================================================================] - 0s

====================================================
load_combo_dose_response : Time = 173.42562413215637 s
====================================================
Loaded 3686475 drug pair dose response measurements
Combined dose response data contains sources: ['CCLE' 'CTRP' 'gCSI' 'GDSC' 'NCI60' 'SCL' 'SCLC' 'ALMANAC.FG'
 'ALMANAC.FF' 'ALMANAC.1A']
====================================================
load_combined_dose_response : Time = 603.7396519184113 s
====================================================
Summary of combined dose response by source:
====================================================
summarize_response_data : Time = 30.22744655609131 s
====================================================
              Growth  Sample  Drug1  Drug2
Source
ALMANAC.1A    208605      60    102    102
ALMANAC.FF   2062098      60     92     71
ALMANAC.FG   1415772      60    100     29
CCLE           93251     504     24      0
CTRP         6171005     887    544      0
GDSC         1894212    1075    249      0
NCI60       18862308      59  52671      0
SCL           301336      65    445      0
SCLC          389510      70    526      0
gCSI           58094     409     16      0
====================================================
encode_sources : Time = 0.021260976791381836 s
====================================================
	 DF Drop 3.0991132259368896 s
	 DF Concat Drop Duplicates DropNa ResetIndex 1.5396080017089844 s
Combined raw dose response data has 3070 unique samples and 53520 unique drugs
Limiting drugs to those with response min <= 1, max >= -1, span >= 0, median_min <= -1, median_max >= 1 ...
====================================================
select_drugs_with_response_range : Time = 7.644067764282227 s
====================================================
Selected 47005 drugs from 53520
====================================================
read_set_from_file : Time = 1.1205673217773438e-05 s
====================================================
====================================================
read_set_from_file : Time = 4.0531158447265625e-06 s
====================================================
	 Read Set From File 5.435943603515625e-05 s
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_rnaseq_data_lincs1000_source_scale
100630528/100635564 [============================>.] - ETA: 0s
Loaded combined RNAseq data: (15198, 943)
====================================================
load_cell_rnaseq : Time = 28.92943525314331 s
====================================================
	 Load Cell rnaseq 28.929776906967163 s
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/drug_info
90112/82273 [================================] - 0s

====================================================
load_drug_info : Time = 0.33562254905700684 s
====================================================
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/Combined_PubChem_dragon7_descriptors.tsv
9568256/9602258 [============================>.] - ETA: 0s
	 CSV Read Time 1.439681053161621 s
	 DF Iloc Time 4.76837158203125e-07 s
	 Dict Conv Time Time 0.0009386539459228516 s
	 CSV Read with DType Cols Time 0.4577667713165283 s
	 Other DF Ops 0.010889530181884766 s
====================================================
load_drug_set_descriptors : Time = 4.766966104507446 s
====================================================
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/NCI60_dragon7_descriptors.tsv
971177984/971172032 [==============================] - 261s

	 CSV Read Time 0.956519603729248 s
	 DF Iloc Time 4.76837158203125e-07 s
	 Dict Conv Time Time 0.0009281635284423828 s
	 CSV Read with DType Cols Time 32.263375997543335 s
	 Other DF Ops 2.2116641998291016 s
====================================================
load_drug_set_descriptors : Time = 296.9688265323639 s
====================================================
	 Other DF Ops [iloc, impute, concat, extraction] 14.797099828720093 s
Loaded combined dragon7 drug descriptors: (53507, 5271)
====================================================
load_drug_descriptors : Time = 316.9221656322479 s
====================================================
====================================================
load_drug_info : Time = 0.012031316757202148 s
====================================================
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/Combined_PubChem_dragon7_PFP.tsv
1032192/1069338 [===========================>..] - ETA: 0s
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/Combined_PubChem_dragon7_ECFP.tsv
1048576/1069355 [============================>.] - ETA: 0s
====================================================
load_drug_set_fingerprints : Time = 1.67649245262146 s
====================================================
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/NCI60_dragon7_PFP.tsv
108773376/108805893 [============================>.] - ETA: 0s
Downloading data from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/NCI60_dragon7_ECFP.tsv
108707840/108805910 [============================>.] - ETA: 0s
====================================================
load_drug_set_fingerprints : Time = 65.52673506736755 s
====================================================
Loaded combined dragon7 drug fingerprints: (53507, 2049)
====================================================
load_drug_fingerprints : Time = 68.74209570884705 s
====================================================
Filtering drug response data...
	 Merge DF 1 cellids 0.0077326297760009766 s
  2375 molecular samples with feature and response data
	 Merge DF 2 drug_ids 0.06518912315368652 s
	 Merge DF 3 merge 0.03043985366821289 s
  46837 selected drugs with feature and response data
	 DF isin/isnull 5.406120538711548 s
Summary of filtered dose response by source:
====================================================
summarize_response_data : Time = 22.24741530418396 s
====================================================
              Growth  Sample  Drug1  Drug2
Source
ALMANAC.1A    206580      60    101    101
ALMANAC.FF   2062098      60     92     71
ALMANAC.FG   1293465      60     98     27
CCLE           80213     474     22      0
CTRP         3397103     812    311      0
GDSC         1022204     672    213      0
NCI60       17190561      59  46272      0
gCSI           50822     357     16      0
====================================================
load_drug_info : Time = 0.0031385421752929688 s
====================================================
Grouped response data by drug_pair: 51763 groups
====================================================
assign_partition_groups : Time = 14.854915618896484 s
====================================================
	 DF assign 16.11030912399292 s
Input features shapes:
  dose1: (1,)
  dose2: (1,)
  cell.rnaseq: (942,)
  drug1.descriptors: (5270,)
  drug1.fingerprints: (2048,)
  drug2.descriptors: (5270,)
  drug2.fingerprints: (2048,)
Total input dimensions: 15580
Saved data to cache: cache/all.pkl
====================================================
CombinedDataLoader.load : Time = 1119.5683727264404 s
====================================================
Combined model:
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input.cell.rnaseq (InputLayer)  (None, 942)          0
__________________________________________________________________________________________________
input.drug1.descriptors (InputL (None, 5270)         0
__________________________________________________________________________________________________
input.drug1.fingerprints (Input (None, 2048)         0
__________________________________________________________________________________________________
input.drug2.descriptors (InputL (None, 5270)         0
__________________________________________________________________________________________________
input.drug2.fingerprints (Input (None, 2048)         0
__________________________________________________________________________________________________
input.dose1 (InputLayer)        (None, 1)            0
__________________________________________________________________________________________________
input.dose2 (InputLayer)        (None, 1)            0
__________________________________________________________________________________________________
cell.rnaseq (Model)             (None, 1000)         2945000     input.cell.rnaseq[0][0]
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         7273000     input.drug1.descriptors[0][0]
                                                                 input.drug2.descriptors[0][0]
__________________________________________________________________________________________________
drug.fingerprints (Model)       (None, 1000)         4051000     input.drug1.fingerprints[0][0]
                                                                 input.drug2.fingerprints[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 5002)         0           input.dose1[0][0]
                                                                 input.dose2[0][0]
                                                                 cell.rnaseq[1][0]
                                                                 drug.descriptors[1][0]
                                                                 drug.fingerprints[1][0]
                                                                 drug.descriptors[2][0]
                                                                 drug.fingerprints[2][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1000)         5003000     concatenate_1[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1000)         1001000     dense_10[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1000)         1001000     dense_11[0][0]
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            1001        dense_12[0][0]
==================================================================================================
Total params: 21,275,001
Trainable params: 21,275,001
Non-trainable params: 0
__________________________________________________________________________________________________
partition:train, rank:0, sharded index size:20158304, batch_size:32, steps:629947
partition:val, rank:0, sharded index size:5144704, batch_size:32, steps:160772
Between random pairs in y_val:
  mse: 0.6069
  mae: 0.5458
  r2: -1.0000
  corr: 0.0000
Data points per epoch: train = 20158304, val = 5144704
Steps per epoch: train = 629947, val = 160772
WARNING:tensorflow:From /home/vibhatha/anaconda3/envs/ENVSCIAPPS/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10
```