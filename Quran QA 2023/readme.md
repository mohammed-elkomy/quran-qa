# Quran QA 20223 shared task B participation

Use the `TCE QQA2023-B demo notebook.ipynb` file to get an overview of:
1. installation
2. data download
3. faithful data generation
4. pre-training data generation
5. training and fine-tuning of a LM
6. downloading and obtaining a dump file

**Once the training is made you will find a dump file saved!**

Something like: bert-base-arabertv02-fine-tuned-2e-05-first-827-QQA23_TaskB_qrcd_v1.2_train.zip
This is a bert-base-arabertv02 fine-tuned model with:
1. learning rate of 2e-05. 
2. first learning method.
3. A random starting seed of 827.
4. QQA23_TaskB_qrcd_v1.2_train training data is used

This dump file contains all summary results and model predictions for each sample. 
You can look at the **analysis** directory of the repo for more details.
You can group dump files into folders:
1. run **performance_analysis.py** script
2. Then run **ensemble_analysis.py** to get ensemble results from the **performance_analysis.py** script results.
3. run **generate_reports.py**  to get excel aggregated results and kernel density plots for different runs. 


## Repo Structure
```
├── analysis
          ├── dataset_basic_analysis.py (fining basic statistics)
          ├── draw_threshold_curve.py  (effect of thresholding analysis)
          ├── ensemble_analysis.py  (script to apply ensemble on all trained models in subfolders)
          ├── generate_reports.py (writing excel reports from dump files)
          ├── no_answer_leakage_analysis.py (leakage analysis for no answer questions)
          ├── performance_analysis.py (script to analyse dump files)
├── answer_decoding (code for answer decoding)
├── answer_list_post_processing (code for post-processing)
├── artifacts (generated dumps, runs and submissions (download this))
├── configs (config scripts for transfomers)
├── data (all generated and downloaed data is here)
├── data_scripts (code for dataloading and generation for pretraining+ faithful splitting)
├── ensemble (code for ensemble)
├── metrics (code for evaluation metrics)
├── models  (code for HF subclassed models)
├── requirements.txt
├── runners  (code for training script)
├── text_processing_helpers (code for helper scripts)
├── trainers (code for HF Trainer)
```
