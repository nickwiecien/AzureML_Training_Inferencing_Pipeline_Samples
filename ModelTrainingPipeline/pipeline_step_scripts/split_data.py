# Step 2. Split Data
# Sample Python script designed to retrieve a pandas dataframe
# containing raw data, then split that into train and test subsets.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
from numpy.random import seed

# Parse input arguments
parser = argparse.ArgumentParser("Split raw data into train/test subsets.")
parser.add_argument('--training_data', dest='training_data', required=True)
parser.add_argument('--testing_data', dest='testing_data', required=True)
parser.add_argument('--testing_size', type=float, required=True)

args, _ = parser.parse_known_args()
training_data = args.training_data
testing_data = args.testing_data
testing_size = args.testing_size

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
raw_datset = current_run.input_datasets['Raw_Data']
raw_df = raw_datset.to_pandas_dataframe().astype(np.float64)

################################# MODIFY #################################

# Optionally include data transformation steps here. These may also be
# included in a separate step entirely.

##########################################################################

# Split into train and test subsets according to the user-provided testing size
train, test = train_test_split(raw_df, test_size=testing_size)

# Save train data to both train and test dataset locations.
os.makedirs(training_data, exist_ok=True)
os.makedirs(testing_data, exist_ok=True)
train.to_csv(os.path.join(training_data, 'training_data.csv'), index=False)
test.to_csv(os.path.join(testing_data, 'testing_data.csv'), index=False)
