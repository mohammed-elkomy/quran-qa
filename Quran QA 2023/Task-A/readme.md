# Quran QA 2023 shared task A participation

Use the `TCE QQA2023-A demo notebook.ipynb` file to get an overview of:
1. installation
2. data download
3. pre-training data generation
4. training and fine-tuning of a LM
5. downloading and obtaining a dump file

🎉 All models can be found on my [huggingface account](https://huggingface.co/MatMulMan)

**Once the training is made you will find a dump file saved!**

something like: araelectra-base-discriminator-tafseer-pairs-fine-tuned-1e-06-5254-train.zip
This is an araelectra-base-discriminator-tafseer-pairs fine-tuned model with:
1. learning rate of 1e-06.
2. A random starting seed of 5254.
4. train.zip means training data is used

This dump file contains models prediction for the given eval or test data.

You can look at the **analysis** directory of the repo for more details.
You can group dump files into folders:
1. run **performance_analysis.py** script to process and get results for single models and ensemble models
   - **retrieval_ensemble.py** is consumed by **performance_analysis.py** to implement the ensemble logic

```
├── analysis
          ├── dataset_basic_analysis.py (fining basic statistics)
          ├── draw_threshold_curve.py  (effect of thresholding analysis)
          ├── perfromance_analysis.py (script to analyse dump files)
          ├── retrieval_ensemble.py (script to apply ensemble on all trained models in subfolders)
├── artifacts ├── artifacts (generated dumps, runs and submissions (download this))
├── configs (Training configs)
├── data (all generated and downloaed data is here)
├── data_scripts (code for dataloading and generation for pretraining)
├── lexical (code for lexical matching baselines)
├── metrics (code for evaluation metrics)
├── readme.md
├── requirements.txt
├── biencoder (A clone of DRhard repo (STAR algorithm))
├── cross_encoder (code for training and inferrence for cross-encoders)
└── sbert (code for training and inferrence for sbert bi-encoders)
```
