# LinearRegresion.tf.keras

Auto MPG Analysis and Modeling
Overview
This program performs data analysis and builds machine learning models to predict the fuel efficiency (miles per gallon, MPG) of cars based on various attributes. The dataset used is the "Auto MPG" dataset from the UCI Machine Learning Repository. The program includes data preprocessing, normalization, and building several regression models, including linear and deep neural network models.

Steps
Importing Libraries: The program imports necessary libraries for data manipulation, visualization, and machine learning, including pandas, numpy, matplotlib, seaborn, and tensorflow.

Loading Data:

The dataset is loaded from a URL.
Column names are defined for easier data manipulation.
Missing values are handled by removing rows with missing entries.

Data Transformation:

The 'Origin' column, which contains categorical data, is transformed into one-hot encoded columns representing different regions (Poland, USA, Japan).
Data Splitting:

The dataset is split into training (80%) and testing (20%) sets to evaluate model performance on unseen data.
Feature and Label Preparation:

Features (input variables) and labels (target variable, MPG) are separated.
All features and labels are converted to float32 type for compatibility with TensorFlow.

Normalization:

A normalization layer is created and adapted to the training data to scale features to a standard range.
Model Building and Training:

Linear Regression Model: Built using TensorFlow to predict MPG based on horsepower. The model is trained and its performance is plotted.
Multi-dimensional Linear Regression Model: A linear model that uses all features to predict MPG.
Deep Neural Network (DNN) Models:
A DNN model using only the 'Horsepower' feature.
A DNN model using all features.
Model Evaluation:

The models are evaluated on the test dataset, and their mean absolute error (MAE) is calculated.
Performance results of all models are summarized and printed.
Predictions and Visualization:

Predictions are made using the DNN model with all features.
The true vs predicted MPG values are plotted.
The distribution of prediction errors is plotted.
Model Saving:

The trained DNN model is saved for future use.
Execution
To execute the program, run the script in a Python environment with the necessary libraries installed. The script performs all steps from data loading to model evaluation and visualization automatically.

python
Copier le code
# Example command to run the script
python regresja_liniowa.py
Output
Model Performance: Mean absolute error (MAE) for each model.
Visualizations:
Loss curves for training and validation sets.
Scatter plot of true vs predicted MPG values.
Histogram of prediction errors.
Saved Model: The trained DNN model is saved as dnn_model.
Dependencies
Ensure the following Python libraries are installed:

matplotlib
numpy
pandas
seaborn
tensorflow
keras
You can install these dependencies using pip:


"pip install matplotlib numpy pandas seaborn tensorflow keras"

This documentation provides a comprehensive overview of the program's functionality, making it easier for others to understand its purpose and usage.
