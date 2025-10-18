# Math5470

This repository contains the course project for **Math5470**, a mathematics course designed to explore fundamental techniques in machine learning.

## 🧭 Project Overview
As part of the course requirement, students work individually or in small teams to complete a hands-on machine learning project. The project involves data analysis, model development, and result interpretation, with emphasis on **scientific reasoning** rather than just performance.

## 📊 Selected Project: M5 Forecasting
Our team chose the **Kaggle M5 Forecasting – Accuracy** competition.

**Objective:**  
To predict daily unit sales of Walmart retail products over a 28-day horizon using hierarchical time series data from multiple stores across California, Texas, and Wisconsin.

**Dataset Features:**  
- Item-level, department, and store-level data  
- Explanatory variables such as prices, promotions, calendar events, and special days  

**Goal:**  
Improve forecast accuracy by combining traditional time-series approaches with modern machine learning techniques.

**Competition link:**  
[Kaggle M5 Forecasting – Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy)

## ⚙️ Environment Setup
Clone this repository and create the Conda environment using the provided `.yml` file:

```bash
git clone <your-repo-link>
cd <your-repo-folder>
conda env create -f environment.yml
conda activate math
```

## 📂 Dataset Download
Download the M5 Forecasting dataset from Kaggle and place it in the following directory structure:

Math5470\
|-- calendar.csv\
|-- sales_train_validation.csv\
|-- sell_prices.csv\
|-- sample_submission.csv

Dataset download link: [🔗 [Link](https://drive.google.com/drive/folders/1hQxd15oEfHEqxxnW7SI9e9harf3J5Ux7?usp=sharing)]

Make sure the files are unzipped and located in the `math5470/` folder before running any training or analysis scripts.

## 🧠 Model Download
Download the pretrained model or checkpoints (if available) and place them in the following directory:

Math5470/ \
|-- model.lgb \
|-- model_meta.json


Model download link: [🔗 [Link](https://drive.google.com/drive/folders/1RUk4qG1w7Y-ryRTMHEFQKKKPVNX20AR4?usp=sharing)]

Ensure that the model file name and path match the configuration in your training or inference scripts.

## 🚀 Training
Run the training script to start model training:

```bash
python train.py
```
Make sure the dataset and environment are properly set up before running this command. Training logs and checkpoints will be automatically saved in the designated output directory.

## 🔍 Inference

After training, use the inference script to generate predictions:
```bash
python infer.py
```

## 📈 Evaluation

Evaluate the model performance using the provided evaluation script:

```bash
python eval.py
```

## ⚡ Train Other Models
We also provide the scripts for training and evaluating a xgboost model. You can follow it and write your own method. Feel free to try it! 
```bash
python train_xgboost.py
python infer_xgboost.py
```


## 👥 Contribution

| Name | Contribution |
|------|---------------|
| Weizhen Bian | Performed initial data cleaning and feature extraction; implemented the main model, including training, inference, and evaluation; and contributed to writing and editing the final report. |
| Yiming Li |  |
| Pengyu Chen |  |
| Jiahao Pan |  |
| Boyi Kang |  |

