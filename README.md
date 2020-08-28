# Titanic-Survival-Prediction
In this repository, we will go through the whole process of creating a machine learning model on the famous _Titanic dataset_. We have two .csv files for the training and testing of data so as to make predictions. Initially, we began with analyzing our training data and checking for any missing data and found out which features are the most significant for making better predictions. During this whole process, we used seaborn and matplotlib libraries to perform the visualizations. During the data preprocessing part, we computed missing values, converted features into numeric ones, grouped values into categories and created a few new features. Afterwards we started training 9 different machine learning models, picked one of them (random forest) and applied cross validation on it. Then we discussed how random forest works, took a look at the importance it assigns to the different features and tuned it’s performace through optimizing it’s hyperparameter values. Lastly, we looked at it’s confusion matrix and computed the models precision, recall and f-score.
