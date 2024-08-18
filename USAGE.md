## For TRAINING:

### Download and process the AIST++ dataset (wavs and motion only) using:
```.bash
cd data
bash download_dataset.sh
python create_dataset.py --extract-baseline
```

### Train your own model
#### (Arguments are not taken from the command line, but from args.py! (parse_train_opt()))
Once the AIST++ dataset is downloaded and processed, run the training script, e.g.
```.bash
python ./train.py
```


## For EVALUATION:
### Run test to generate and save features from tracks (taken from custom_music)
#### (Arguments are not taken from the command line, but from args.py! (parse_test_opt()))
``` .bash
python ./pipeline.py
```

Make sure to point the arguments in args.py to the checkpoint you want to test with.