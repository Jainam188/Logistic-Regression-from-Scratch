#!/usr/bin/env python
# coding: utf-8

# Imported All Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Reading CSV file using pandas and Printing of Head of the dataset

# In[2]:


train = pd.read_csv('adult-training.csv')
test = pd.read_csv('adult-test.csv', skiprows=1)

print(train.head())


# We have some ? in our dataset so Replacing that by NAN value and finding and summing of all null values in dataset.

# In[3]:


train.replace(' ?', np.nan, inplace=True)
test.replace(' ?', np.nan, inplace=True)

print(train.isnull().sum())


# In[4]:


print(test.isnull().sum())


# Feature Engineering
# 
# Selecting Income as a Taget Value.
# Converting >50K as 1 and <=50 by 0.

# In[5]:


train['Income'] = train['Income'].apply(lambda x: 1 if x==' >50K' else 0)
test['Income'] = test['Income'].apply(lambda x: 0 if x==' >50K' else 0)


# Filling Not Applicable Values by 0 in Workclass column and printing the head of train data.

# In[6]:


train['Workclass'].fillna(' 0', inplace=True)
test['Workclass'].fillna(' 0', inplace=True)
print(train.head(20))


# Plotting the Features with Target Value Using catplot as Barchart. 

# Plotting Workclass feature column with Target value that is Income.

# In[7]:


sns.catplot(x='Workclass', y='Income', data=train, kind='bar', height=6)
plt.xticks(rotation=45);
plt.show()


# Counting Values in Workclass Column

# In[8]:


train['Workclass'].value_counts()
test['Workclass'].value_counts()


# As we can see in above Never-worked and Without-pay are indicating single purpose
# Merge without-pay and Never-worked.

# In[9]:


train['Workclass'].replace(' Without-pay', 'Never-worked', inplace=True)
test['Workclass'].replace(' Without-pay', 'Never-worked', inplace=True)


# Describing the fnlgwt column

# In[10]:


train['fnlgwt'].describe()


# As we can see above that fnlght column has high mean and standard deviation. so we are applying logarithm for reducing mean and standard deviation.

# In[11]:


train['fnlgwt'] = train['fnlgwt'].apply(lambda x: np.log1p(x))
test['fnlgwt'] = test['fnlgwt'].apply(lambda x: np.log1p(x))

print(train['fnlgwt'].describe())


# Plotting Education column with Target Value Income

# In[12]:


sns.catplot(x='Education', y='Income', data=train, kind='bar', height=7)
plt.xticks(rotation=60)
plt.show()


# There are too many categories are available so we are combining some of categories in single category as primary.

# In[13]:


def primary(x):
    if x in ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'Primary'
    else:
        return x


train['Education'] = train['Education'].apply(primary)
test['Education'] = test['Education'].apply(primary)


# Re plotting the Eduction, Income chart after above changes. 

# In[14]:


sns.catplot(x='Education', y='Income', data=train, height=6, kind='bar')
plt.xticks(rotation=60)
plt.show()


# Plotting Marital Status columns with Target Column Income.

# In[15]:


sns.catplot(x='Marital Status', y='Income', height=5, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# Counting all Categories in Marital Status.

# In[16]:


train['Marital Status'].value_counts()


# Merging Married-AF-spouse category into Married-civ-spouse.

# In[17]:


train['Marital Status'].replace('Married-AF-spouse', 'Married-civ-spouse', inplace=True)
test['Marital Status'].replace('Married-AF-spouse', 'Marries-civ-spouse', inplace=True)


# Replotting After Merging

# In[18]:


sns.catplot(x='Marital Status', y='Income', height=5, kind='bar', data=train, palette='muted')
plt.xticks(rotation=60)
plt.show()


# Filling NA values by 0 in Occupation column.

# In[19]:


train['Occupation'].fillna(' 0', inplace=True)
test['Occupation'].fillna(' 0', inplace=True)


# plotting Occupation Feature column with Target Value Income

# In[20]:


sns.catplot(x='Occupation', y='Income', height=8, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# Counting Categories Values of Occupation column

# In[21]:


train['Occupation'].value_counts()


# As we can see above Armed-Forces has only 9 values so it cannot needed so we are removing that by merging it values with 0.
# and also re-plotting.

# In[22]:


train['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)
test['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)

sns.catplot(x='Occupation', y='Income', height=8, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# Plotting Relationship feature column with Target Value Income.

# In[23]:


sns.catplot(x='Relationship', y='Income', height=6, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# In[24]:


train['Relationship'].value_counts()


# Plotting Race column with Income

# In[25]:


sns.catplot(x='Race', y='Income', height=8, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# In[26]:


train['Race'].value_counts()


# Plotting Sex Column

# In[27]:


sns.catplot(x='Sex', y='Income', height=8, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# Filling NA values in Native Country by 0

# In[28]:


train['Native country'].fillna(' 0', inplace=True)
test['Native country'].fillna(' 0', inplace=True)


# Plotting Native Country by Target Value Income

# In[29]:


sns.catplot(x='Native country', y='Income', height=10, kind='bar', data=train)
plt.xticks(rotation=80)
plt.show()


# As we can see above that Native Country Column has to many countries are there, for reducing the countries we are dividing all countries into regions as shown below and replotting Native Country column.

# In[30]:


def native(country):
    if country in [' United-States', ' Cuba', ' 0']:
        return 'US'
    elif country in [' England', ' Germany', ' Canada', ' Italy', ' France', ' Greece', ' Philippines']:
        return 'Western'
    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', ' Columbia', ' Laos', ' Portugal', ' Haiti',
                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru',
                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Vietnam',
                     ' Holand-Netherlands']:
        return 'Poor'  # no offence
    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', ' Japan', ' Yugoslavia', ' China', ' Hong']:
        return 'Eastern'
    elif country in [' South', ' Poland', ' Ireland', ' Hungary', ' Scotland', ' Thailand', ' Ecuador']:
        return 'Poland team'

    else:
        return country


train['Native country'] = train['Native country'].apply(native)
test['Native country'] = test['Native country'].apply(native)

sns.catplot(x='Native country', y='Income', height=5, kind='bar', data=train)
plt.xticks(rotation=60)
plt.show()


# Joint Both train and test dataset

# In[31]:


joint = pd.concat([train,test], axis=0)
joint.dtypes


# Selecting only object columns from dataset

# In[32]:


categorical_features = joint.select_dtypes(include=['object']).axes[1]
for col in categorical_features:
    print (col, joint[col].nunique())


# Splitting all Categories of every columns as a single column and splitting by :

# In[33]:


for col in categorical_features:
    joint = pd.concat([joint, pd.get_dummies(joint[col], prefix=col, prefix_sep=':')], axis=1)
    joint.drop(col, axis=1, inplace=True)

joint.head()


# Taking Train and Test data
# Xtrain are all feature columns
# Ytrain is Target Value
# same ofr test data

# In[34]:


train = joint.head(train.shape[0])
test = joint.tail(test.shape[0])

Xtrain = train.drop('Income', axis=1)
Ytrain = train['Income']

Xtest = test.drop('Income', axis=1)
Ytest = test['Income']


# Creating Logistic Regression from scratch
# defining funtions:
# Hypothesis using Sigmoid function
# Calculating Loss function
# Fit
# Predict

# In[42]:


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    #Hypothesis
    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    #Loss Function
    def __loss(self, h, y):
        return (-y * np.log(h) - (1-y) * np.log(1-h)).mean()

    #fitting values
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h-y)) / y.size
            self.theta -= self.lr * gradient

    #Probability of prediction
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    #predicting the value
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


# After this we can use Logistic Regression as we were using in Sklearn lib
# 1. fitting values in model
# 2. predicting the values
# 3. Final Accuracy

# In[46]:


model = LogisticRegression()
model.fit(Xtrain, Ytrain)

Ztrain = model.predict(Xtrain)
Ztest = model.predict(Xtest)


# In[47]:


print('Accuracy of the Logistic Regression Model', (Ztrain == Ytrain).mean())

