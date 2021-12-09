# Step 4. Evaluate and Register Model
# Sample Python script designed to evaluate a newly-trained
# challenger model against a previously-trained "champion"
# model (i.e., the best-performing model to date). In the case
# that the challenger performs better, it should be registered in the
# workspace, otherwise the run should end gracefully.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
import shutil
import sklearn
import joblib
import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser("Evaluate classified and register if more performant")
parser.add_argument('--training_outputs', type=str, required=True)
parser.add_argument('--target_column', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_description', type=str, required=True)

args, _ = parser.parse_known_args()
training_outputs = args.training_outputs
target_column = args.target_column
model_name = args.model_name
model_description = args.model_description

# Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

# Get training/testing datasets
# Both are associated with registered models.
training_dataset = current_run.input_datasets['Training_Data']
testing_dataset = current_run.input_datasets['Testing_Data']
formatted_datasets = [('Training_Data', training_dataset), ('Testing_Data', testing_dataset)]

# Get point to PipelineData from previous training step
training_step_pipeline_data = training_outputs

# Copy training outputs to relative path for registration
relative_model_path = 'model_files'
current_run.upload_folder(name=relative_model_path, path=training_step_pipeline_data)

# Load test dataset
X_test_dataset = current_run.input_datasets['Testing_Data']
X_test = X_test_dataset.to_pandas_dataframe().astype(np.float64)

# Split into X and y components
y_test = X_test[[target_column]]
X_test = X_test.drop(target_column, axis=1)

################################# MODIFY #################################

# The intent of this block is to load the newly-trained challenger model
# and current champion model (if it exists), and evaluate performance of
# both against a common test dataset using a target metric of interest.
# If the challenger performs better, it should be added to the model
# registry.

# Load challenger scaler and model
scaler = joblib.load(os.path.join(training_step_pipeline_data, 'scaler.pkl'))
knn = joblib.load(os.path.join(training_step_pipeline_data, 'model.pkl'))

# Scale data using challenger scaler
challenger_X_test = scaler.transform(X_test)

# Get current model from workspace
model_list = Model.list(ws, name=model_name, latest=True)
first_registration = len(model_list)==0

# Calculate challenger metrics
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, roc_auc_score, recall_score, f1_score
challenger_preds = knn.predict(challenger_X_test)
challenger_accuracy = accuracy_score(y_test, challenger_preds)

# Create tag dictionary
updated_tags = {'Challenger Accuracy': challenger_accuracy, 'Champion Accuracy':'N/A'}

#If no model exists register the current model
if first_registration:
    print('First model registration.')
    model = current_run.register_model(model_name, model_path='model_files', description=model_description, model_framework='Scikit-Learn', model_framework_version=sklearn.__version__, tags=updated_tags, datasets=formatted_datasets, sample_input_dataset = training_dataset)
    current_run.set_tags(updated_tags)
else:
    # If a model has been registered previously, check to see if challenger model 
    # performs better (higher accuracy). If so, register it.
    Model(ws, name=model_name).download(target_dir='champion_outputs', exist_ok=True)
    champion_model = joblib.load('champion_outputs/model_files/model.pkl')
    champion_scaler = joblib.load('champion_outputs/model_files/scaler.pkl')
    champion_X_test = champion_scaler.transform(X_test)
    champion_preds = champion_model.predict(champion_X_test)
    champion_accuracy = accuracy_score(y_test, champion_preds)
    updated_tags['Champion Accuracy'] = champion_accuracy
    current_run.set_tags(updated_tags)
    if champion_accuracy < challenger_accuracy:
        print('New model performs better than existing model. Register it.')
        model = current_run.register_model(model_name, model_path='model_files', description=model_description, model_framework='Scikit-Learn', model_framework_version=sklearn.__version__, tags=updated_tags, datasets=formatted_datasets, sample_input_dataset = training_dataset)
    else:
        print('New model does not perform better than existing model. Cancel run.')
        current_run.parent.cancel()
        
##########################################################################