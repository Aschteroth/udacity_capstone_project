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

*Fig. 1: AutoMl settings*

### Results

My AutoML experiment returned a StackEnsemble as best performing model, with a weighted AUC score of 0.88999. 
A "StackEnsemble", "Stacked Generalization" or "Stacking", is 
  
  "an ensemble machine learning Algorithm, that learns how to best combine the predictions from multiple well-performing machine learning models." [Source](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)



![image](https://user-images.githubusercontent.com/61315167/118228220-183bf080-b48a-11eb-8525-ce8c65af1b3b.png)

*Fig. 2: Run Detail Widget showing completion*

![image](https://user-images.githubusercontent.com/61315167/118228280-31dd3800-b48a-11eb-93a4-a37f2fa7ccf6.png)

*Fig. 3: The best models obtained in the experiment in descending order*

![image](https://user-images.githubusercontent.com/61315167/118228334-4cafac80-b48a-11eb-98d6-0b17cbcf992a.png)

*Fig. 4: The best model, a StackEnsemble*


The score of 0.88999 is already pretty good, but there are some options to improve that score. 

I could tune the experiments parameters and apply data cleaning and data munging techniques. For example
- Giving the model more time to run by increasing the timeout duration. The experiment could run more models, but this will incur higher cost. 
- Choosing another primary metric than AUC weighted
- Collecting more data for the dataset. Since the last entry is from 2017, we could get a little bit more than 3 years worth of additional data. 
- Try different startegies to handle missing data like imputation, dropping them altogether or using random values in the range of data for each column
- One-hot encoding categorical data

## Hyperparameter Tuning

For the hyperparameter experiment, I chose a [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/tree.html) model from the sklearn-library. A decision tree is a non-parametric supervised learning method used for classification and regression that will predict a label column by learning from the data features.

I decided to use config parameters similar to the parameters used in the first project of this course: 
- Early termination improves computational efficiency, but might return a slightly worse result by missing some good candidates. I chose the "Bandit" policy, an aggressive policy based on slack factor/slack amount and evaluation interval, that early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run. I specified a slack factor of 0.1 and an evaluation interval of 3.[Source and further reading](https://azure.github.io/azureml-sdk-for-r/reference/bandit_policy.html)
- Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. Compared to a Grid search, where we try every combination of a preset list of values of the hyper-parameters and evaluate the model for each combination, Random search yields equal or even better results with comparably less resources. Since the provided lab is limited to 4 hours, I chose the faster RandomParameterSampling. [Source and further reading](https://medium.com/@senapati.dipak97/grid-search-vs-random-search-d34c92946318#:~:text=Random%20search%20works%20best%20for,are%20less%20number%20of%20dimensions)

![image](https://user-images.githubusercontent.com/61315167/118266783-5e5d7800-b4bb-11eb-9cd3-2d40cc892ce7.png)

*Fig. 5: hyperdrive config*


### Results

The best run from the hyperdrive experiments scored a weighted AUC of 0.86481, the randomly chosen parameters were criterion:gini, splitter: best and a max depth of 6. The run took 5 minutes and 48 seconds to complete. Since the result was slightly below the result from the AutoMl model, I chose to discard the HyperDrive models.


![image](https://user-images.githubusercontent.com/61315167/118231393-5be52900-b48f-11eb-9bf0-6a73ec6432d6.png)

*Fig. 6: Run Detail Widget showing completion of hyperparameter experiment*

![image](https://user-images.githubusercontent.com/61315167/118231494-86cf7d00-b48f-11eb-9b22-a854b19f397e.png)


*Fig. 7: Performance of the experiments child runs*

![image](https://user-images.githubusercontent.com/61315167/118231427-6b647200-b48f-11eb-9144-fd09dd538037.png)

*Fig. 8: The highest scoring run of the experiment *

#### Improvements
There are some options one could explore to improve the results from the HyperDrive experiment:

- Try another algorithm. There are numerous other classification algorithms to choose from, e.g. Linear Support Vector Machines, XGBoost, Light Gradient Boosting Machines or Random Forests.
- Choose another sampling method. A grid search is more exhaustive but might yield better results. Bayesian sampling will probably yield the best results, but is even more computationally exhaustive than a grid search.
- Tune or change the early termination policy. I could experiment with stricter slack factors and different evaluation intervals or use another policy or drop it altogether. 
- Increase the number of maximum total runs

[Further reading](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)

## Model Deployment

I deployed the best performing model of the AutoML experiment as a web service. To deploy the model in Azure, I had to follow 3 steps: 

1) Register the model in my workspace. The screenshot below shows the code for the registration process. 

![image](https://user-images.githubusercontent.com/61315167/118364599-5b38b980-b599-11eb-8ae0-c03100774dbe.png)
*Fig. 9: Registering a model*

2) Register an image including the model, its scoring script and environment file in a docker container.
The scoring script is generated when a model is created. It provides the code to run for the image, while the environment file contains all the dependencies we need to run the script.

See [Microsoft docs](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.inferenceconfig?view=azure-ml-py) for more details.

![image](https://user-images.githubusercontent.com/61315167/118364677-c4203180-b599-11eb-9a8f-5496ba64c7aa.png)
*Fig. 10: Downloading the necessary files and configuring the Webservice*

3) Deploy the created image as a web service

![image](https://user-images.githubusercontent.com/61315167/118365053-64c32100-b59b-11eb-82a6-d7f3020b45dd.png)
*Fig. 11: Deployment*

![image](https://user-images.githubusercontent.com/61315167/118365714-49a5e080-b59e-11eb-90a0-c1c24786bd1e.png)
*Fig. 12: The deployed web service, showing "healthy" status*

## Querying the service

The deployed service takes an HTTP POST request submitting data in JSON format, similar to a python dictionary with following keys as input: 
{"data": [{"Date": str,
                "Location": str,
                "MinTemp": float,
                "MaxTemp": float,
                "Rainfall": float,
                "Evaporation": str,
                "Sunshine": str,
                "WindGustDir": str,
                "WindGustSpeed": float,
                "WindDir9am": str,
                "WindDir3pm": str,
                "WindSpeed9am": float,
                "WindSpeed3pm": float,
                "Humidity9am": float,
                "Humidity3pm": float,
                "Pressure9am": float,
                "Pressure3pm": float,
                "Cloud9am": float,
                "Cloud3pm": float,
                "Temp9am": float,
                "Temp3pm": float,
                "RainToday": boolean}] 

Where every key corresponds to a column in the underlying dataset. 
The response will either be "true" or "false", where "true" means, it will rain tomorrow in Australia. 

![image](https://user-images.githubusercontent.com/61315167/118365759-8245ba00-b59e-11eb-94cc-c85857ccf842.png)

*Fig. 13: Sampling test data from our dataset*

![image](https://user-images.githubusercontent.com/61315167/118365850-a6090000-b59e-11eb-8fdd-c86e6dfbf47a.png)

*Fig. 14: Sending the request*

## Screen Recording
You can take a look at my project on youtube [here](https://www.youtube.com/watch?v=34cjqPEEy1M)
