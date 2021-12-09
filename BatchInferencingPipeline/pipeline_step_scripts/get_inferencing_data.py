# Step 1. Get Inferencing Data
# Sample Python script designed to load data from a target data source,
# and export as a tabular dataset

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser("Get Inferencing Data")
parser.add_argument('--inferencing_dataset', dest='inferencing_dataset', required=True)

args, _ = parser.parse_known_args()
inferencing_dataset = args.inferencing_dataset

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Get default datastore
ds = ws.get_default_datastore()

################################# MODIFY #################################

# The intent of this block is to load from a target data source. This
# can be from an AML-linked datastore or a separate data source accessed
# using a different SDK. Any initial formatting operations can be be 
# performed here as well.

# Read all raw data from blob storage & convert to a pandas data frame
csv_paths = [(ds, 'iris_data_scoring/*')]
score_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
score_df = score_ds.to_pandas_dataframe().astype(np.float64)

##########################################################################

# Save dataset for consumption in next pipeline step
os.makedirs(inferencing_dataset, exist_ok=True)
score_df.to_csv(os.path.join(inferencing_dataset, 'inferencing_data.csv'), index=False)