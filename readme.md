# ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists

![](./logo/ascends.png)

ASCENDS is a toolkit that is developed to assist scientists or any persons who want to use their data for machine leearning.
We know that there are so many tools available for data scientists, but most of them require programming skills and often overwhelming.
We aim to provide a set of simple but powerful tools for non-data scientists to be able to intuitively perform various 
advanced data analysis and machine learning techniques with simple interfaces (**a command-line interface** and a web-based GUI).

The current version of ASCENDS mainly focuses on two different machine learning tasks - classification and regression (value prediction). 

* What is classification?
Users can train a predictive model (mapping function) that predicts a category (Y) from input variables (X) 
using ASCENDS. For instance, ASCENDS can train a model for predicting whether an email is spam or not-spam. 

* What is regression?
Users can train a predictive model (mapping function) that approximates a continuous output variable (y) from input variables (X) 
using ASCENDS. For instance, ASCENDS can train a model for predicting a value of house based on square footage, 
number of bedrooms, number of cars that can be parked in its garages, number of storages.

ASCENDS current version is 0.2.2, and we are currently beta-testing the software.

ASCENDS principles

- Supporting various classification/regression techniques (Linear regression, logistic regression, random forest, support vector machine, neural network, ...) 
- Supporting Feature selection based on various criteria
- Provides automatic hyperparameter tuning
- No programming skills required; but ASCEBDS library can be used in your code if needed
- Using standard CSV (comma separated values) format data set 
- Built on top of open source projects (keras, tensorflow, scikit-learn, etc.)

Although ASCENDS has been originally developed for material scientists' research on high temperature alloy design,
the tool can be also used for many other applications.

We encourage you to cite using the following BibTex citation, if you have used our tool:
```
@misc{snapnets,
  author       = {Sangkeun Lee and Dongwon Shin and Jian Peng},
  title        = {{ASCENDS}: A data SCiENce toolkit for Non-Data Scientists},
  howpublished = {\url{https://code.ornl.gov/slz/ascends-toolkit}},
  month        = jan,
  year         = 2019
}
```
List of ORNL contributors
* Sangkeun Lee, Core Developer (lees4@ornl.gov, leesangkeun@gmail.com)
* Dongwon Shin, (shind@ornl.gov)
* Jian Peng (pengj@ornl.gov)


# Installation (With Anaconda)

ASCENDS requires Python version 3.X, and using Anaconda is recommended.
Please download Anaconda from https://www.anaconda.com/download/ and install first.

Once you installed Anaconda (with Python 3.X), you can create a new Python environment for ascends by doing:

```
conda create --name ascends
```

To activate an environment:
- On Windows, in your Anaconda Prompt, run 
```
activate ascends
```
- On macOS and Linux, in your Terminal Window, run
```
source activate ascends
```

You will see the active environment in parentheses at the beginning of your command prompt:
```
(ascends) $
```
Please install pip in your local conda environment by doing:
```
(ascends) $ conda install --yes pip
```

Then, install ascends-toolkit by doing:
```
(ascends) $ pip install ascends-toolkit
```

Now you're ready to use ascends. Please see the next section for a quick start guide.
To check if you properly installed ascends-toolkit, run 
```
(ascends) $ train_regression.py -h
```
If you see the usage help of regression trainer, you're ready to go.
Now have fun with ascends-toolkit.

To deactivate the current Anaconda environment, after using ascends.

- On Windows, in your Anaconda Prompt, run 
```
deactivate ascends
```
- On macOS and Linux, in your Terminal Window, run
```
source deactivate ascends
```

# Getting Started: Classification

To train your first model, we are going to use the Iris data set, which is one of the classic datasets introduced by the British statistician and biologist Ronald Fisher in his 1936 paper. 
Download the iris.csv into your data directory. (Save as the following link)

Link: https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv

The Iris data set consists of 50 samples from each of 3 species of Iris (Iris setosa, Iris virginica and Iris versicolor). 
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Using ASCENDS, we will train a model that can predict a species of Iris, when new input data is given.

ASCENDS uses standard CSV (Comma Separated Values) file format,
and the file requires to have a header in the first line. The following shows the first 5 lines of the Iris data set.

```
SepalLength,SepalWidth,PetalLength,PetalWidth,Name
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
```
Let's try to train a classification model by executing the following command to train a classification model using ASENDS:
In this tutorial, we assume tat you already created a output directory to save output files and stored data files in data directory.

```train_classifier.py data/iris.csv output/iris Name --mapping "{'Name': {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}}```

`data/iris.csv` is the input file that we just downloaded in the data folder. `output/iris` is the path and the tag that will be used for output files. 
`Name` is the target column name. So, ASCENDS will train a model that predicts `Name` when four other column (SepalLength,SepalWidth,PetalLength,PetalWidth) 
values are given. As we can see in the first 5 lines of data file, values for the column `Name` is not numerical. For training, we need to map
the categorical values into numerical values. This is done by using `--mapping` option. The example command will map `Iris-setosa` to 0, `Iris-versicolor` to 1, and `Iris-virginica` to 2
 for all values of column `Name`.

Then you will see, the following result.

```
$ train_classifier.py data/iris.csv output/iris Name --mapping "{'Name': {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}}"
Using TensorFlow backend.

 * ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists 
 * Classifier ML model trainer 

 programmed by Matt Sangkeun Lee (lees4@ornl.gov) 

 [ Data Loading ]
 Loading data from :data/iris.csv
 Columns to ignore :None
 Input columns :['SepalLength' 'SepalWidth' 'PetalLength' 'PetalWidth']
 Target column :Name
 Using default scikit-learn hyperparameters 
 Overriding parameters from command-line arguments ..

 The following parameters will be used: 

 [ Model Evaluation ]
* (RF)	 accuracy =    0.947 5-fold cross validation 
 Saving tuned hyperparameters to file:  data/iris.csv,Model=RF,Scaler=StandardScaler.tuned.prop

 [ Model Save ]
* Training initiated ..
* Training done.
* Trained model saved to file: ./output/iris,Model=RF,accuracy=0.9466666666666667,Scaler=StandardScaler.pkl
```

When training is done. ASCENDS shows an expected accuracy that is calculated via 5-fold cross validation.
The result says that expected accuracy is 94.7% (not bad!). The trained model is saved as a file (~.pkl)
So, now let's see how the trained model file can be used for classification 
with unknown (data that has not been used for training) input data.

Copy the following text into a new file and save it to `data\iris_test_input.csv`.

```
SepalLength,SepalWidth,PetalLength,PetalWidth
7.2,2.5,4.1,1.3
5.2,5.5,4.1,1.3
```

As we can see above, we don't know what class each line belongs. Let's run the following:

```
classify_with_model.py output/iris\,Model\=RF\,accuracy\=0.9466666666666667\,Scaler\=StandardScaler.pkl data/iris_test_input.csv output/iris_test_prediction.csv --mapping "{'Name': {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}}"
```

Executing the above command will predict category for the input data `data\iris_test_input.csv` and result will be saved in `output/iris_test_prediction.csv`.
Note that we specified the trained model file we achieved ealier via ```train_classifier.py``` command.

When you open up the generated output file `output/iris_test_prediction.csv`,
```
,SepalLength,SepalWidth,PetalLength,PetalWidth,Name
0,7.2,2.5,4.1,1.3,Iris-versicolor
1,5.2,5.5,4.1,1.3,Iris-versicolor
```
We can see that the model thanks that both are Iris-versicolor.

# Getting Started: Regression

Let's have some fun with regression. We are going to use Boston Housing Data. Download the file from following link and save in `data/BostonHousing.csv`.
Link: https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. 
The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository
- crim: per capita crime rate by town 
- zn: proportion of residential land zoned for lots over 25,000 sq.ft. 
- indus: proportion of non-retail business acres per town 
- chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
- nox: nitric oxides concentration (parts per 10 million) 
- rm: average number of rooms per dwelling 
- age: proportion of owner-occupied units built prior to 1940 
- dis: weighted distances to ﬁve Boston employment centers 
- rad: index of accessibility to radial highways 
- tax: full-value property-tax rate per $10,000 
- ptratio: pupil-teacher ratio by town 
- b: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 
- lstat: % lower status of the population 
- medv: Median value of owner-occupied homes in $1000s

The following shows how to train a regression model using ASCENDS to predict `medv`.

```
$ train_regression.py data/BostonHousing.csv output/BostonHousing medv
Using TensorFlow backend.

 * ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists 
 * Regression (value prediction) ML model trainer 

 programmed by Matt Sangkeun Lee (lees4@ornl.gov) 

 [ Data Loading ]
 Loading data from :data/BostonHousing.csv
 Columns to ignore :None
 Input columns :['crim' 'zn' 'indus' 'chas' 'nox' 'rm' 'age' 'dis' 'rad' 'tax' 'ptratio'
 'b' 'lstat']
 Target column :medv
 Using default scikit-learn hyperparameters 
 Overriding parameters from command-line arguments ..

 The following parameters will be used: 
{'scaler_option': 'StandardScaler', 'rf_n_estimators': '100', 'rf_max_features': 'auto', 'rf_max_depth': 'None', 'rf_min_samples_split': '2', 'rf_min_samples_leaf': '1', 'rf_bootstrap': 'True', 'rf_criterion': 'mse', 'rf_min_weight_fraction_leaf': '0.', 'rf_max_leaf_nodes': 'None', 'rf_min_impurity_decrease': '0.', 'nn_n_neighbors': '5', 'nn_weights': 'uniform', 'nn_algorithm': 'auto', 'nn_leaf_size': '30', 'nn_p': '2', 'nn_metric': 'minkowski', 'nn_metric_params': 'None', 'kr_alpha': '1', 'kr_kernel': 'linear', 'kr_gamma': 'None', 'kr_degree': '3', 'kr_coef0': '1', 'br_n_iter': '300', 'br_alpha_1': '1.2e-6', 'br_alpha_2': '1.e-6', 'br_tol': '1.e-3', 'br_lambda_1': '1.e-6', 'br_lambda_2': '1.e-6', 'br_compute_score': 'False', 'br_fit_intercept': 'True', 'br_normalize': 'False', 'svm_kernel': 'rbf', 'svm_degree': '3', 'svm_coef0': '0.0', 'svm_tol': '1e-3', 'svm_c': '1.0', 'svm_epsilon': '0.1', 'svm_shrinking': 'True', 'svm_gamma': 'auto', 'net_structure': '16 16 16', 'net_layer_n': '3', 'net_dropout': '0.0', 'net_l_2': '0.01', 'net_learning_rate': '0.01', 'net_epochs': '100', 'net_batch_size': '2'}

 [ Model Evaluation ]
 Saving test charts to :  output/BostonHousing,Model=RF,MAE=2.271302387431676,R2=0.8571003609225923,Scaler=StandardScaler.png
* (RF)	 MAE =    2.271, R2 =    0.857 via 5-fold cross validation 
 Saving tuned hyperparameters to file:  data/BostonHousing.csv,Model=RF,Scaler=StandardScaler.tuned.prop

 [ Model Save ]
* Training initiated ..
* Training done.
* Trained model saved to file: ./output/BostonHousing,Model=RF,MAE=2.271302387431676,R2=0.8571003609225923,Scaler=StandardScaler.pkl
```
Expected MAE (Mean Absolute Error) is 2.271 and R2 (https://bit.ly/2pP83Eb) is 0.857.

Copy the following text into a new file and save it to `data/BostonHousing_test_input.csv`.

```
crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat
0.00532,15,1.31,"0",0.538,5.575,62.1,4.09,1,296,15.3,396.9,4.98
0.02231,0,7.07,"0",0.469,5.421,78.9,3.9671,2,242,14.8,396.9,8.14
```

Similar to classification example, let's run the following:
```
regression_with_model.py ./output/BostonHousing,Model=RF,MAE=2.271302387431676,R2=0.8571003609225923,Scaler=StandardScaler.pkl data/BostonHousing_test_input.csv output/BostonHousing_test_prediction.csv
```

Executing the above command will predict category for the input data `data/BostonHousing_test_input.csv` and result will be saved in `output/BostonHousing_test_prediction.csv`.
Note that we specified the trained model file we achieved ealier via ```train_regression.py``` command.

# License

MIT License
Please contact us at lees4@ornl.gov, shind@ornl.gov for more details
