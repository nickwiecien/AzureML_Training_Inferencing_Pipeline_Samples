# Step 3. Train Model
# Sample Python script designed to train a K-Neighbors classification
# model using the Scikit-Learn library.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse
import shutil

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, roc_auc_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import neighbors
import matplotlib.pyplot as plt
import joblib
from numpy.random import seed


# Parse input arguments
parser = argparse.ArgumentParser("Train classification model")
parser.add_argument('--train_to_evaluate_pipeline_data', dest='train_to_evaluate_pipeline_data', required=True)
parser.add_argument('--target_column', type=str, required=True)

args, _ = parser.parse_known_args()
train_to_evaluate_pipeline_data = args.train_to_evaluate_pipeline_data
target_column = args.target_column

# Get current run
current_run = Run.get_context()

#G et associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
X_train_dataset = current_run.input_datasets['Training_Data']
X_train = X_train_dataset.to_pandas_dataframe().astype(np.float64)
X_test_dataset = current_run.input_datasets['Testing_Data']
X_test = X_test_dataset.to_pandas_dataframe().astype(np.float64)

# Split into X and y 
y_train = X_train[[target_column]]
y_test = X_test[[target_column]]

X_train = X_train.drop(target_column, axis=1)
X_test = X_test.drop(target_column, axis=1)

################################# MODIFY #################################

# The intent of this block is to scale data appropriately and train
# a predictive model. Any normalizaton and training approach can be used.
# Serialized scalers/models can be passed forward to subsequent pipeline
# steps as PipelineData using the syntax below. Additionally, for 
# record-keeping, it is recommended to log performance metrics 
# into the current run.

# Normalize input data data
scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                      columns=X_train.columns,
                      index=X_train.index)

# Train classification model
knn=neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)

# Save model to outputs for record-keeping
os.makedirs('./outputs', exist_ok=True)
from joblib import dump, load
dump(knn, './outputs/model.pkl')
dump(scaler, './outputs/scaler.pkl')

# Save model to pipeline_data for use in evaluation/registration step
os.makedirs(train_to_evaluate_pipeline_data, exist_ok=True)
dump(knn, os.path.join(train_to_evaluate_pipeline_data, 'model.pkl'))
dump(scaler, os.path.join(train_to_evaluate_pipeline_data, 'scaler.pkl'))

# Generate predictions
X_test = scaler.transform(X_test)
preds = knn.predict(X_test)

# Add metrics to the current run
tag_dict = {'Framework':'scikit-learn', 'Model Type': 'K-Neighbors Classifier'}
current_run.set_tags(tag_dict)

# Create a confusion matrix and log in the run
cmatrix = confusion_matrix(y_test, preds)
cmatrix_json = {
    "schema_type": "confusion_matrix",
       "schema_version": "v1",
       "data": {
           "class_labels": ["0", "1", "2"],
           "matrix": [
               [int(x) for x in cmatrix[0]],
               [int(x) for x in cmatrix[1]],
               [int(x) for x in cmatrix[2]]
           ]
       }
}

current_run.log_confusion_matrix('ConfusionMatrix_Test', cmatrix_json)

# Calculate accuracy when scoring the training dataset and log in the run
current_run.log('Accuracy', accuracy_score(y_test, preds))

##########################################################################