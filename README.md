# Azure ML Model Training/Evaluation & Batch Inferencing Pipeline Samples

Sample notebooks demonstrating how to create a model training pipeline and batch inferencing pipeline using the Azure ML SDK. 

The model training pipeline `(./ModelTrainingPipeline/AML_CreateTrainingPipeline.ipynb)` loads raw data from an [AML-linked datastore](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data), registers the raw dataset, splits and registers test and train datasets, trains a classification model, and evaluates and registers the model and associated scalers. In this final step, the sample notebook showcases how to establish an A/B testing pattern for champion and challenger (current best and newly-trained) models so that the best performer is always reflected in the model registry.

The batch inferencing pipeline `(./BatchInferencingPipeline/AML_CreateBatchInferencingPipeline.ipynb)` loads unscored data from an AML-linked datastore, loads the best performing model, scores the data, and saves the data to a new time-stamped location in the associated datastore.

For demonstration purposes, we leverage the [Iris Setosa dataset from Scikit-Learn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) to train a basic classification model and make predictions. We have included code in the notebooks described above to automatically load this dataset into your default AML datastore so it can be consumed by the sample pipelines.

## Environment Setup
<b>Note:</b> Recommend running these notebooks on an Azure Machine Learning Compute Instance using the preconfigured `Python 3.6 - AzureML` environment.

To build and run the sample pipelines contained in `./ModelTrainingPipeline` and `./BatchInferencingPipeline` the following resources are required:
* Azure Machine Learning Workspace