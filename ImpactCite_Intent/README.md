# ImpactCite Intent Analysis

**Required sources**

- XLNet GitHub link: https://github.com/zihangdai/xlnet
- SciCite Github link: https://github.com/allenai/scicite (only data needed)
- Pretrained Model link: https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip

**Intent data**

Following commands are used to train and test the model:

**Step 1:** Download and unzip the required sources in this directory.

The dataset should be in the *scicite_data* folder. The files of the pretrained model should be in the *xlnet_cased_L-24_H-1024_A-16* folder and the XLNet git should be directly in this directory.

**Step 2:** Prepare the data:

`python Train_.py`

**Step 3:** Finetune the pretrained model:

`python run_classifier.py --do_train=True --do_eval=True --do_predict=True --predict_dir=xlnet_output --eval_all_ckpt=True --task_name=imdb --data_dir=aclImdb --output_dir=xlnet_output --model_dir=xlnet_checkpoint --uncased=False --spiece_model_file=xlnet_cased_L-24_H-1024_A-16/spiece.model --model_config_path=xlnet_cased_L-24_H-1024_A-16/xlnet_config.json --init_checkpoint=xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt --max_seq_length=256 --train_batch_size=2 --eval_batch_size=2 --num_hosts=1 --num_core_per_host=1 --learning_rate=2e-5 --train_steps=4000 --warmup_steps=500 --save_steps=500 --iterations=500`

**Step 24** Test the model:

`python Tester.py`

## Checkpoint files + code
https://cloud.dfki.de/owncloud/index.php/s/aS72TdD47iDPmJf
