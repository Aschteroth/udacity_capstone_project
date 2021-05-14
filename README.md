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

I decided to stick with the automl-settings that we used in the previous excercise.

- The timeout is set to 20 minutes for efficiency reasons
- the number of max concurrent iterations is set to 5, one below the maximum number of nodes
- I chose AUC_weighted (Area under the curve) as primary metric which is specific to binary classification problems. 

My choices for the config were as follows:

- Since we want to answer the question "Will it rain in Australia tomorrow?" where the answer is of a binary nature, either "Yes" or "No", we are dealing with a classification problem.
- Our target column is "RainTomorrow"
- Since this is just a project to show my newly aquired skills, I enabled early stopping for efficiency reasons. Otherwise, the model might run for several hours or even days without improving much.

![image](https://user-images.githubusercontent.com/61315167/118227684-3fde8900-b489-11eb-833e-dd684afbc213.png)
*Fig. 1: AutoMl

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

My AutoML experiment returned a StackEnsemble as best performing model, with a weighted AUC score of 0.88999. 
A "StackEnsemble", "Stacked Generalization" or "Stacking", is 
  "an ensemble machine learning Algorithm, that learns how to best combine the predictions from multiple well-performing machine learning models." 
  
Source: [machinelearningmastery.com](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)



![image](https://user-images.githubusercontent.com/61315167/118228220-183bf080-b48a-11eb-8525-ce8c65af1b3b.png)

*Fig. 2: Run Detail Widget showing completion*

![image](https://user-images.githubusercontent.com/61315167/118228280-31dd3800-b48a-11eb-93a4-a37f2fa7ccf6.png)

*Fig. 3: The best models obtained in the experiment in descending order*

![image](https://user-images.githubusercontent.com/61315167/118228334-4cafac80-b48a-11eb-98d6-0b17cbcf992a.png)

*Fig. 4: The best model, a StackEnsemble*


The score of 0.88999 is already pretty good, but there are some options to improve that score: 
- Giving the model more time to run by increasing the timeout duration. The experiment could run more models, but this will incur higher cost. 
- Choosing another primary metric than AUC weighted
- Collecting more data for the dataset. Since the last entry is from 2017, we could get a little bit more than 3 years worth of additional data. 
- Try different startegies to handle missing data like imputation, dropping them altogether or using random values in the range of data for each column


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

*Fig. 5: Run Detail Widget showing completion of hyperparameter experiment*


*Fig. 6: The best models obtained in the experiment in descending order*


*Fig. 7: The best model, a *

The score of 0.88999 is already pretty good, but there are some options to improve that score: 


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.



## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
You can take a look at my project [here](https://www.youtube.com/watch?v=34cjqPEEy1M)
