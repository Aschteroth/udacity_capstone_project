# Machine learning Engineer with Microsoft Azure - Capstone Project
## a Udacity Nanodegree

In this final project I will set up a workspace and compute cluster in Microsoft AzureMl studio and work with a free dataset from an external source, creating an  Automated Machine Learning (AutoMl) and a Hyperdrive experiment with the Python SDK. The best performing model will then be deployed and can be consumed via web service.

## Dataset

For this capstone project, I will use a [Dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package?select=weatherAUS.csv) compiled by Joe Young, originally hosted on kaggle.com and uploaded to my personal [github account](https://github.com/Aschteroth/udacity_capstone_project).

The Dataset contains 10 years worth of meteorological data (years 2007-2017), organised in 23 columns with 145461 rows. THe data was collected in different places in Australia, including the info if it rained that day and if it rained the day after.

## Task

Given a specific set of weather parameters like location, temperature, wind direction, evaporation and humidity, the web service is meant to predict if there will be rain tomorrow in Australia. The underlying problem is a binary classification problem where the outcome can either be "True" - it will rain -  or "False" - there will be no rain.

"Classification is a type of supervised learning in which models learn using training data, and apply those learnings to new data. [...] The main goal of classification models is to predict which categories new data will fall into based on learnings from its training data."
[Source](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)

### Access
The dataset is hosted on my personal github account and accessed via URL with azureMLs [*Dataset* class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py). 
The data is then converted to a pandas dataframe.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
You can take a look at my project [here](https://www.youtube.com/watch?v=34cjqPEEy1M)
