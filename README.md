# Osteoporosis Detector Using Explainable AI

>  [!Note]
> This is a deep learning generative AI model that will produce a classification of Osteoporotic stage from analyzing an X-ray of a patient's femur.

## Overview

Osteoporosis is the gradual deterioration of bone structure with time. As age increases, vital vitamins and minerals needed to maintain bone health decrease, leading to weak and brittle bones that are easily susceptible to fractures.

Osteoporosis, especially for hip bones are classified based off the Singh Index

![Trabecular Patterns by Singh Index](Misc/Trabecular_Singh.png "Trabecular Patterns by Singh Index")

Predicting early osteoporosis is vital. We aim to cover this by developing a screening tool with the help of Machine Learning. 

We aim to utilize dual methodologies for detecting.

•⁠  ⁠An efficient ensemble stacking utilizing Optuna for hyperparameter search

•⁠  U-Net for image segmentation to crop out the hip portion of the x-ray.

•⁠  ⁠YOLOv7 CNN for X-ray classification based off Singh Index.

![Architecture Diagram](Misc/Architecture_Diagram.png "Architecture Diagram")

## Execution
1) Navigate to the home directory and run the command to download the dependencies(you must be in Python 3.10):
   ```bash
   pip3 install -r requirements.txt
   ```
2) Navigate into the (`/UI/`) directory and run the command:
   ```bash
   streamlit run app.py
   ```
