This repo was originally cloned and modified from a snapshot of the official Sail-On-API reposotory similar to https://github.com/darpa-sail-on/Sail-On-API/tree/umd_svo

# Novel-Snapshot Serengeti Server API

## Setup

Insure python3 is accessible and pipenv is ibtalled.

1. From the ss-api' directory, run 'pipenv --python 3.8'
2. Run 'pipenv install'
2. Run 'pipenv shell'
3. Run 'python setup.py install'

All other commands should be run within the pipenv shell or pipenv envronment.

### Running the  ss-api

To run the API locally, use the following command:
```bash
cd ~/ss-api/
sail_on_server_ss --url 127.0.0.1:8005 \
--data-directory '/test_trials/api_tests/' \
--bboxes-json-file 'test.json' \
--results-directory 'Experiments/Exp_1_EWC'
```

### Running the [DCA system](https://github.com/guyera/ss-osu-system/) client

`git clone https://github.com/guyera/ss-osu-system/`
```bash
cd ~/ss-osu-system
torchrun \
   --nnodes=1 \
   --nproc_per_node=4 \
   --rdzv_id=103 \
   --rdzv_endpoint=localhost:28319 \
   main.py \
   --detection-feedback \
   --url 'http://127.0.0.1:8005' \
   --trial-size 3000 \
   --trial-batch-size 10 \
   --test_ids OND.102.000 OND.103.000 OND.104.000 OND.105.000 OND.1061.000 OND.1062.000 OND.1063.000 OND.1064.000 \
   --root-cache-dir .data-cache \
   --train-csv-path /nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.csv \
   --pretrained-models-dir 'pretrained-models-balanced-normalization-corrected/train-heads/hack' \
   --precomputed-feature-dir '.features/hack/resizedpad=224/none/normalized' \
   --classifier-trainer ewc-train \
   --retraining-lr 1e-5 \
   --retraining-batch-size 64 \
   --retraining-max-epochs 50 \
   --gan_augment False \
   --distributed \
   --feedback-sampling-configuration combined \
   --feedback-loss-weight 0.5 \
   --should-log \
   --log-dir 'Experiments/Exp_1_EWC/logs'
```


## Data Generation/Evalutation Service

`sail_on_server --url localhost:3306 --data-directory ./tests/data --results-directory ./test/results`


### Provider

This is an interface that can be extended per project needs if the given FileProvider as described below isn't sufficient
Implementing your own provider requires:
* extension of sail_on.api.Provider
* main routine, using main() and command_line() of sail_on/api/server.py as a basis.

### FileProvider
This tools provides FileProvider. The FileProvider serves all data generation function needs.
It provide a very basic scoring capability.  It is designed to be either
 (1) server as an example on how to provide the data generation or (2) to be basis for extension to its capability.
and evaluation services

The FileProvider assumes tests for protocols have been pre-constructed. This includes:
* List of Tests per Protocol and Domain
* Contents of Tests per Protocol and Domain
* Ground Truth for Tests per Protocol and Domain 

The FileProvider is given two directories: a location for the test data and location to store test results.
The file structure for test data is:
+ PROTOCOL LEVEL -> each folder is named after the PROTOCOL (e.g. OSND)
+ + DOMIN LEVEL -> each folder is named after a domain (e.g. images)
+ + + test_ids.csv -> a file summarizing all TEST ID files in the same folder
+ + + TEST DATA FILE: <PROTOCOL>.<TEST>.<NO>.<SEED>.csv files contain the list of examples by URI or filename
+ + + TEST LABEL FILE:<PROTOCOL>.<TEST>.<NO>.<SEED>_<GTTYPE>.csv files contain the list of examples by URI or filename along the ground truth labels for the specific type of ground truth.  There may be more than one type (e.g. classification, characterization).

### Configure Hints for UMD_SVO
Hint Type B : No changes needed. One can request for Type B hint for given detection only.
Hint Type A : Create a csv file with two columns : ['test_id' , 'novelty_kind']. Server will read this csv and will return the corresponding hint for the test_id. Name the csv file "hints_info.csv" and put that file in the same location as your test_files ( {path}/OND/svo_classification/hints_info.csv )

## KEY COMPONENTS

* SEED is useful for distinguishing different seeds used for each test set.
* TEST is a name or number used to group files all designed to for one test to attain statistical signifiance.
* NO is the incremental test set number.
* URI or filename assumes that information is reachable by the client (system under test)
* GT CSV files have two mandatory columns: the image file URI or filename (as matched to the test data file).  All other columns are reserved for scoring.
These never shared with the client via the protocol.


## CAUTION

* At no time should the same test file contents change.  It is better to create a new file with a new name.
We want to allow the client system to pre-cache and pre-processing the test data given the URI or filename  (e.g. place in a pyTorch dataset).


# WSGI

Running with WSGI requires a configfile (python) to specify the bind port for the service AND the worker processes count:

```
import multiprocessing

bind = "0.0.0.0:5000"
workers = 1 #multiprocessing.cpu_count() * 2 + 1
```

Run wsgi in the PIPENV environment providing the config file for WSGI and the location of the test and results directories:

```
LOG_NAME=`date +"%m-%d-%Y.%H.%M"`
pipenv run gunicorn -c gunicorn.config.py 'sail_on.wsgi:create_app(data_directory="/home/robertsone/TESTS", results_directory="/home/robertsone/RESULTS")' >> "${LOG_NAME}_unicorn_0.txt" 2>&1
```

## License
This repo is a product of collaboration between multiple entities, and different portions of the source code are licensed under different terms. Please familiarize yourself with `DISCLAIMER`, `LICENSE`, and any copyright notices documented in the source code.
