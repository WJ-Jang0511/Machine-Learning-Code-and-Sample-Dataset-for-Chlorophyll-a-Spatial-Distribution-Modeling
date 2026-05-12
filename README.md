Description
This repository contains the sample dataset, pre-trained model, and Python source codes required to reproduce the machine learning workflow presented in the manuscript submitted to Ecological Informatics.

Contents of the Repository:

Sample_Input.csv: A sample dataset containing extracted band ratio values and Chlorophyll-a concentrations for model training and validation.

1_Imbalance_Code.py: Python script for data resampling to address class imbalance issues (e.g., using SMOTEENN, ADASYN).

2_Train_ML.py: Python script for training various machine learning and artificial neural network models, including hyperparameter tuning.

3_Spartial_Distribution_Chl-a.py: Python script that applies the best-performing model to Sentinel-2 imagery to generate spatial distribution maps.

Pre-trained Model: The repository includes a sample pre-trained model (3CL_SMOTEENN_GB.pkl) generated from the workflow.

Spatial Data: Includes a base raster (basemap.tif) and shapefiles (Namyang.shp, etc.) representing the study area boundary used for clipping and spatial analysis.

Usage:
To demonstrate the methodology, execute the scripts sequentially (1 -> 2 -> 3). Ensure that the required spatial data and sample inputs are located in the same working directory as the scripts.
