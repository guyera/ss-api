This repo was originally cloned and modified from a snapshot of the official Sail-On-API reposotory similar to https://github.com/darpa-sail-on/Sail-On-API/tree/umd_svo

# Novel-Snapshot Serengeti Server API

## Setup

From the ss-api' directory, Run 

`python setup.py install`


### Running the  ss-api

To run the API locally, use the following command:
```bash
sail_on_server_ss --url 127.0.0.1:8005 \
--data-directory '/test_trials/api_tests/' \
--bboxes-json-file 'test.json' \
--results-directory 'Experiments/Exp_1_EWC'
```

### Running the [DCA system](https://github.com/guyera/ss-osu-system/) client

`git clone https://github.com/guyera/ss-osu-system/`
```bash
torchrun \
   --nnodes=1 \
   --nproc_per_node=1 \
   --rdzv_id=103 \
   --rdzv_endpoint=localhost:28319 \
   main.py \
   --detection-feedback \
   --url 'http://127.0.0.1:8005' \
   --trial-size 3000 \
   --trial-batch-size 10 \
   --test_ids OND.102.000 OND.103.000 \
   --root-cache-dir data-cache \
   --train-csv-path sail_on3/final/osu_train_cal_val/train.csv \
   --pretrained-models-dir pretrained-models \
   --precomputed-feature-dir 'precomputed-features/resizedpad=224/none/normalized' \
   --classifier-trainer ewc-train \
   --retraining-lr 1e-5 \
   --retraining-batch-size 64 \
   --retraining-max-epochs 50 \
   --gan_augment False \
   --distributed \
   --feedback-loss-weight 0.5 \
   --should-log \
   --log-dir 'logs'
```


## License
This repo is a product of collaboration between multiple entities, and different portions of the source code are licensed under different terms. Please familiarize yourself with `DISCLAIMER`, `LICENSE`, and any copyright notices documented in the source code.
