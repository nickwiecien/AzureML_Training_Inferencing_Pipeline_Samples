# Step 3. Publish Scored Data
# Sample Python script designed to save scored data into a 
# target (sink) datastore.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Get default datastore
ds = ws.get_default_datastore()

# Get inferencing dataset
scored_dataset = current_run.input_datasets['scored_data']
scored_data_df = scored_dataset.to_pandas_dataframe()

################################# MODIFY #################################

# You can optionally save data to a separate datastore in this step.

##########################################################################

# Save dataset to ./outputs dir
os.makedirs('./outputs', exist_ok=True)
scored_data_df.to_csv(os.path.join('outputs', 'scored_data.csv'), index=False)