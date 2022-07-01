# TCE submission

This is our code submission for qrcd please contact me if you encounter any issue
```mohammed.a.elkomy@gmail.com```
```mohammed.a.elkomy@f-eng.tanta.edu.eg```

# Contents

```
.
├── answer_voting_ensemble.py # the script used for ensemble
├── data # this folder holds the dataset and scripts for reading/ evaluating it
├── post_processing
│         ├── __init__.py
│         ├── print_results_table.py # view checkpoints tables
│         └── results
│             ├── eval # the checkpoints and run files for development phase
│             └── test # the checkpoints and run files for test phase
├── readme.md 
├── run_qa.py # to train a model
├── tmp # a placeholder folder for temp files created
│         └── tmp.md
├── trainer_qa.py # a helper script
└── utils_qa.py # a helper script
```

# How it works

1. I train different models on colab and download a file called .dump file which makes it easier to work with it locally for testing and debugging
2. those dump files represent the results of a trained checkpoint, any file ending with .dump is an example of them
3. to load any file of them just use the library joblib and do ```joblib.load(filename)```
4. after collecting those dump files for all of our trained models we feed them to the ensemble script, we employ a self-ensemble approach where we combine checkpoints of the same model initialized with different seeds.

# reproducing results

```QRCD_demo.ipynb```is demo notebook for reproducing all of the reported checkpoints, you just need to download the dump file generated from colab into your local machine.

1. our ensemble takes a list of dump files to combine them based on their softmax scores
2. then it does post-processing
3. the saved checkpoints will be saved to ```post_processing/results```
4. to reproduce a model when training you need to feed this parameter ```--seed 14``` for a seed of 14
5. if you are trying to reproduce it manually please verify the dump files you download from colab are the same as the ones shared on the drive link above

## eval checkpoints

reproducing table 2 in the paper, total_num_of_models =15LARGE + 15BASE+ 15ARBERT = 45

### bert-large-arabertv02_1

we have 15 checkpoints, seeds are:
```8045, 32558, 79727 ,30429 ,48910 ,46840 ,24384 ,55067 ,13718 ,16213 ,63304 ,40732 ,38609 ,22228 ,71549```

### bert-base-arabertv02

we have 15 checkpoints, seeds are:
```71338 ,67981 ,29808 ,67961 ,25668 ,20181 ,20178 ,67985 ,67982 ,23415 ,20172 ,20166 ,25982 ,27073 ,26612```

### ARBERT

we have 15 checkpoints, seeds are:
```64976 ,64988 ,73862 ,84804 ,79583 ,81181 ,59377 ,59382 ,73869 ,77564 ,79723 ,64952 ,73865 ,59373 ,84349```

## test checkpoints

reproducing table 3 in the paper, total_num_of_models =16LARGE + 18BASE+ 17ARBERT = 51

### bert-large-arabertv02

we have 16 checkpoints, seeds are:
```1114 ,18695 ,23293 ,27892 ,5748 ,59131 ,63847 ,68498 ,73133 ,77793 ,82431 ,87062 ,91701 ,94452 ,96475 ,98797```

### bert-base-arabertv02

we have 18 checkpoints, seeds are:
```54235 ,60998 ,64662 ,80936 ,80955 ,80959 ,80970 ,80988 ,82916 ,84448 ,84481 ,84665 ,84749 ,84871 ,87891 ,87917 ,88329 ,88469```

### ARBERT

we have 17 checkpoints, seeds are:
```107 ,14 ,43919 ,47360 ,50798 ,57621 ,86829 ,88813 ,90781 ,91496 ,91533 ,94949 ,95000 ,96521 ,96552 ,98412 ,98465```

## Using the scripts:

1. to train any of the checkpoints above just check ```QRCD_demo.ipynb``` notebook, it runs ```run_qa.py``` script
2. download dump files created while training and write them to ```post_processing/results/eval``` or ```post_processing/results/test```
3. to run the ensemble just run ```python answer_voting_ensemble.py```
    - this will write the files for both the eval and test phase ensemble
    - json submissions files will be saved to ```post_processing/results/eval``` and ```post_processing/results/test``` and
    - if you would like to evaluate them you may use the official  ```quranqa22_eval.py``` script
4. to print the tables reproducing table 2 in the paper just run
    - ```python print_results_table.py```
    - make sure to be in ```post_processing``` directory
5. to evaluate json files, you may run the script ```evaluate_official.py```