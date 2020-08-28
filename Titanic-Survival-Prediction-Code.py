#!/usr/bin/env python
# coding: utf-8

# In[1]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv('C:\\Users\\meghna\\Desktop\\My_Project\\Titanic Survival Prediction\\train.csv')
test_data = pd.read_csv('C:\\Users\\meghna\\Desktop\\My_Project\\Titanic Survival Prediction\\test.csv')


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.shape


# In[6]:


train_data.info()


# In[7]:


train_data.describe()


# #### Above stats show that 38% out of the training-set survived the Titanic. We can also see that the passenger ages range from 0.4 to 80. Also, we can already detect some features, that contain missing values, like the ‘Age’ feature.

# ### Visualize and Handling Missing Values

# In[8]:


train_data.isnull()


# In[9]:


train_data.isnull().sum()


# In[10]:


#percentage of null values in each column
print((train_data.isnull().sum()/891)*100)


# In[11]:


sns.heatmap(train_data.isnull(),yticklabels=True)


# #### As we can see in above heatmap shows Age has some white bars shows that their are approx 20% missing values and Cabin has approx 77% null values which is huge number.

# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train_data)


# #### Above Diagram depicts that the approx 600 were not survived and approx 360 survived

# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='Sex',hue='Survived',data=train_data)


# #### Above diagram depicts that Male Passenger died much as compared to Female and Female Survived much as compared to Male Passenger

# In[14]:


sns.set_style('whitegrid')
sns.barplot(x='Pclass', y='Survived', data=train_data)


# #### Above diagram depicts that Passenger having class 3 died much and Passenger having class 1 survived larger than both class

# In[15]:


FacetGrid = sns.FacetGrid(train_data, row='Embarked', height=3.5, aspect=1.8)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# #### Embarked and Pclass both seem to be correlated with survival, depending on the gender. Women on port Q and on port S have a higher chance of survival. Men are more safe at port C and less safe at port Q and S.

# In[16]:


sns.distplot(train_data['Age'].dropna(),kde=False,color='darkviolet',bins=40)


# #### As we can see in above diagram, Age from 20 to 40 was there at titanic and less number of people are age between 70 to 80 

# In[17]:


train_data['Age'].hist(bins=50,color='black',alpha=0.6)


# #### Above histogram counts the number of occurences in age column

# In[18]:


sns.countplot(x='SibSp',data=train_data)


# #### As above diagram depicts, approx 600 having sibling or spouse is 0 and approx 200 have sibling or spouse is 1 as on...

# In[19]:


train_data['Fare'].hist(color='green',figsize=(6,5),bins=30,alpha=0.8)


# #### Above diagram depicts that most of the passengers had tickets under $100

# In[20]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',palette='autumn',data=train_data)


# #### As above diagram depicts, Passenger class 1 having mean age is approx 36-37 and Passenger class 2 having mean age is around 28-29 and Passenger class 3 having mean age around 24-25

# ## Data Cleaning & Data Preprocessing

# #### Deleting PassangerID fom training data(train_data) because it doesn't contribute to a person's survival probability but we'll still need it in our test_data.

# In[21]:


train_data = train_data.drop(['PassengerId'], axis=1)


# ### Missing Data - 
# #### From the above heatmap and stats, we can see that we have missing values in three features - Cabin(687), Age(177) & Embark(2)
# ### 1. Cabin

# In[22]:


test_data['Cabin'].unique()


# In[23]:


#Below, we are creating a 'Deck' column and extracting the first letter of the cabin code(e.g., C of C85 OR C123) as the first letter denotes the passenger's deck
train_data['Deck'] = train_data['Cabin'].astype(str).str[0]
test_data['Deck'] = test_data['Cabin'].astype(str).str[0]

#Dropping the 'Cabin' column as now we have received our relevant information out of it.
train_data = train_data.drop(['Cabin'], axis=1)
test_data = test_data.drop(['Cabin'], axis=1)

#Replacing str values(letters) into int values(numbers) so that our model is suitable for ML modeling 
train_data.replace({'Deck' : {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "n" : 0}}, inplace=True)
test_data.replace({'Deck' : {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "n" : 0}}, inplace=True)

#'n' here has been extracted out of 'nan' values of cabin hence we replace all the nan by 0


# In[24]:


train_data['Deck'].unique()


# In[25]:


train_data.head(10)


# In[26]:


test_data.head(10)


# ### 2. Age

# #### Imputation of Age

# In[27]:


def impute_age(cols):
    age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


# #### Now, the above function 'impute_age' is defined which will take age and PClass columns and perform mentioned operations, i.e., as we noticed earlier in our first heatmap that our dataset has a lot of null values in age column, that is why, here we are replacing the null values by the mean value in that particular PClass which we observed through our boxplot.

# In[28]:


train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[29]:


train_data['Age'].head()


# In[30]:


sns.heatmap(train_data.isnull(),yticklabels=True)


# #### Now, we can see in the above heatmap that there is no null value in the 'age' column as it is being replaced by a mean value due to imputation based on pclass.

# ### 3. Embarked

# In[31]:


train_data['Embarked'].describe()


# In[32]:


common_val = 'S'
data = [train_data, test_data]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_val)


# #### Here we'll be replacing the null values in the 'Embarked' column with the most common value 'S' which we can observe from the above information

# In[33]:


train_data.head()


# In[34]:


train_data.info()


# #### No null values in our train dataset

# In[35]:


test_data.info()


# In[36]:


test_data['Fare'].isnull().sum()


# #### We can notice one null value in 'Fare' column so we'll replace it with the mean of fair

# In[37]:


test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)


# In[38]:


test_data['Fare'].isnull().sum()


# In[39]:


train_data.head()


# In[40]:


test_data.head()


# #### Next we will be converting columns 'Age' and 'Fare' into round off integer values

# In[41]:


train_data['Age'] = train_data['Age'].round(0).astype(int)
train_data['Fare'] = train_data['Fare'].round(0).astype(int)


# In[42]:


test_data['Age'] = test_data['Age'].round(0).astype(int)
test_data['Fare'] = test_data['Fare'].round(0).astype(int)


# In[43]:


train_data[['Age','Fare']].head()


# In[44]:


test_data[['Age','Fare']].head()


# In[45]:


train_sex = pd.get_dummies(train_data['Sex'],drop_first=True)
train_embark = pd.get_dummies(train_data['Embarked'],drop_first=True)

test_sex = pd.get_dummies(test_data['Sex'],drop_first=True)
test_embark = pd.get_dummies(test_data['Embarked'],drop_first=True)


# #### Next we will delete columns 'Name' and 'Ticket' as name doesn't affect the survival of a person and Ticket has 600+ unique values, hence we will drop these two columns

# In[46]:


train_data = train_data.drop(['Name','Ticket','Sex','Embarked'], axis=1)
test_data = test_data.drop(['Name', 'Ticket','Sex','Embarked'], axis=1)


# In[47]:


train_data.head()


# In[48]:


test_data.head()


# In[49]:


train_data = pd.concat([train_data,train_sex,train_embark],axis=1)

test_data = pd.concat([test_data,test_sex,test_embark],axis=1)


# In[50]:


train_data.head()


# In[51]:


test_data.head()


# In[52]:


train_data.dropna(inplace=True)
train_data.reset_index(drop=True)
train_data.head()


# In[53]:


test_data.dropna(inplace=True)
test_data.reset_index(drop=True)
test_data.head()


# ## Building ML Models

# ##### Now we will train different Machine Learning models as our testing data doesn't have input labels hence we have to compare several ML models and choose the accurate one for our dataset.

# In[54]:


X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test = test_data.drop("PassengerId", axis=1).copy()


# #### Linear Regression

# In[55]:


lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

lr.score(X_train, Y_train)
acc_lr = round(lr.score(X_train, Y_train) * 100, 2)

print(acc_lr,'%')


# #### Stochastic Gradient Descent (SGD) -

# In[56]:


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print(acc_sgd,'%')


# #### Random Forest -

# In[57]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest,'%')


# #### Logistic Regression -

# In[58]:


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(acc_log,'%')


# #### K Nearest Neighbor -

# In[59]:


knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn,'%')


# #### Gaussian Naive Bayes -

# In[60]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian,'%')


# #### Perceptron -

# In[61]:


perceptron = Perceptron(max_iter=10)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron,'%')


# #### Linear Support Vector Machine -

# In[62]:


linear_svc = LinearSVC(dual=False)
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(acc_linear_svc,'%')


# #### Decision tree -

# In[63]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(acc_decision_tree,'%')


# ## Choosing the best model

# In[125]:


results = pd.DataFrame({
    'Model': ['Linear Regression','Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_lr, acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_data = results.sort_values(by='Score', ascending=False)
result_data = result_data.set_index('Score')
result_data.head(9)


# ## K-Fold Cross Validation

# In[65]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# #### Our data inaccuracy can range anywhere between +3.3 to -3.3

# In[124]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# ## Feature Importance

# In[66]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[67]:


importances.plot.bar()


# ## Conclusion
# ##### Embark(Q & S) and Parch doesn’t play a significant role in our random forest classifiers prediction process. Because of that I will drop them from the dataset and train the classifier again. We could also remove more or less features, but this would need a more detailed investigation of the features effect on our model. But I think it’s just fine to remove only Embark and Parch.

# In[68]:


train_data  = train_data.drop("S", axis=1)
train_data  = train_data.drop("Q", axis=1)
train_data  = train_data.drop("Parch", axis=1)

test_data  = test_data.drop("S", axis=1)
test_data  = test_data.drop("Q", axis=1)
test_data  = test_data.drop("Parch", axis=1)


# ### Training Random Forest Again

# In[116]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_predict = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# #### Our random forest model predicts as good as it did before. A general rule is that, the more features you have, the more likely your model will suffer from overfitting and vice versa. But I think our data looks fine for now and hasn't too much features. 
# #### There is also another way to evaluate a random-forest classifier, which is probably much more accurate than the score we used before. What I am talking about is the out-of-bag samples to estimate the generalization accuracy. I will not go into details here about how it works. Just note that out-of-bag estimate is as accurate as using a test set of the same size as the training set. Therefore, using the out-of-bag error estimate removes the need for a set aside test set.

# In[70]:


print("oob score:",round(random_forest.oob_score_, 4)*100, "%")


# #### Now the above data is ready to tune

# ## Hyperparameter Tuning

# In[71]:


param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
clf.fit(X_train, Y_train)
clf.best_params_


# In[ ]:


#### The above operations on the cell takes about 1-2 hr to run on a normal pc so be patient.


# ### Test New Parameters

# In[121]:


# Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 4,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)


random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# ## Further Evaluation

# ### Confusion Matrix

# In[123]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# #### The above confusion matrix shows that we get 476 survived cases correctly and 73 survived cases incorrectly predicted. Also, 100 unsurvived cases incorrectly predicted and 242 unsurvived cases correctly predicted in the training dataset. 

# In[115]:


train_data.Survived.value_counts()


# ### Precision and Recall

# In[74]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))


# ### F score

# In[75]:


from sklearn.metrics import f1_score
f1_score(Y_train, predictions)


# ### Precision Recall Curve

# In[76]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# ### ROC AUC Curve

# In[77]:


from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# ### ROC AUC Score

# In[78]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score: ",r_a_score*100,"%")


# ## Predicted Dataframe

# In[79]:


print(Y_prediction)


# In[80]:


test_Passenger_ID = test_data['PassengerId']
Y_P = sr = pd.Series(Y_prediction)
Predicted_data = pd.concat([test_Passenger_ID,Y_P],axis=1)


# In[81]:


Predicted_data.head(20)


# In[81]: 


Predicted_data.to_csv('Predicted_dataset.csv')


# #### Predicted data is saved in 'Predicted_dataset'
