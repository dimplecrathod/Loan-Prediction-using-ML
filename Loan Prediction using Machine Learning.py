#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction using Machine Learning

# Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

# It is a classification problem where we have to predict whether a loan would be approved or not. In a classification problem, we have to predict discrete values based on a given set of independent variable(s). Classification can be of two types:
# 
# Binary Classification : In this classification we have to predict either of the two given classes. For example: classifying the gender as male or female, predicting the result as win or loss, etc.
# Multiclass Classification : Here we have to classify the data into three or more classes. For example: classifying a movie's genre as comedy, action or romantic, classify fruits as oranges, apples, or pears, etc.

# What is hypothesis generation?
# This is a very important stage in any data science/machine learning pipeline. It involves understanding the problem in detail by brainstorming as many factors as possible which can impact the outcome. It is done by understanding the problem statement thoroughly and before looking at the data.
# 
# Below are some of the factors which I think can affect the Loan Approval (dependent variable for this loan prediction problem):
# 
# Salary: Applicants with high income should have more chances of loan approval.
# Previous history: Applicants who have repayed their previous debts should have higher chances of loan approval.
# Loan amount: Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
# Loan term: Loan for less time period and less amount should have higher chances of approval.
# EMI: Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.

# Data
# Variable
# 
# Description
# 
# Loan_ID
# 
# Unique Loan ID
# 
# Gender
# 
# Male/ Female
# 
# Married
# 
# Applicant married (Y/N)
# 
# Dependents
# 
# Number of dependents
# 
# Education
# 
# Applicant Education (Graduate/ Under Graduate)
# 
# Self_Employed
# 
# Self employed (Y/N)
# 
# ApplicantIncome
# 
# Applicant income
# 
# CoapplicantIncome
# 
# Coapplicant income
# 
# LoanAmount
# 
# Loan amount in thousands
# 
# Loan_Amount_Term
# 
# Term of loan in months
# 
# Credit_History
# 
# credit history meets guidelines
# 
# Property_Area
# 
# Urban/ Semi Urban/ Rural
# 
# Loan_Status
# 
# Loan approved (Y/N)

# In[315]:


#1. Importing Packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[316]:


# 2. Reading Data
train = pd.read_csv("train_loan.csv")
test = pd.read_csv("test_loan.csv")
train.head()


# In[317]:


train.columns
test.columns

#We have 12 dependent variables and 1 independent variable called Loan_status

train.shape, test.shape


# In[318]:


# 3. Data Analysis
# a. Univariate Analysis -  For categorical features we can use frequency table or bar plots which will calculate the number of each category in a particular variable. For numerical features, probability density plots can be used to look at the distribution of the variable.

# Target Variable - Loan_Status is a categorical variable.

train['Loan_Status'].value_counts()


# In[319]:


# Normalize can be set to True to print proportions instead of number
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()

#The loan of 422 people out of 614 was approved. 


# Variables in the dataset
# Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
# Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
# Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

# In[320]:


#Independent Variable (Categorical)
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(40,20), title= 'Gender') 
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()

#We infer the following:
# 1. 80% applicants in dataset are male.
# 2. 65% are married
# 3. 15% are self-employed.
# 4. 85% have repaid their debts


# In[321]:


# Ordinal Independent Variables
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title='Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()

#We infer the following from the bar plots:
#1. Most applicants don't have any dependents.
#2. 80% are graduates.
#3. Most of applicants are from semiurban area


# In[322]:


# Independent Variable (Numerical)

# 1. Applicant Income
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()

# It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed. We will try to make it normal in later sections as algorithms works better if the data is normally distributed.

# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different education levels.


# In[323]:


# 2. Education
train.boxplot(column='ApplicantIncome', by = 'Education') 
#We can see that there are a higher number of graduates with very high incomes, which are appearing to be the outliers.


# In[324]:


# 3. CoApplicant Income Distribution
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()

# We see a similar distribution as that of the applicant income. Majority of coapplicant’s income ranges from 0 to 5000. We also see a lot of outliers in the coapplicant income and it is not normally distributed.


# In[325]:


plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df['LoanAmount']);
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()

# We see a lot of outliers in this variable and the distribution is fairly normal. 


# In[229]:


#Hypotheses
# Applicants with high income should have more chances of loan approval.
# Applicants who have repaid their previous debts should have higher chances of loan approval.
# Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
# Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.

#The pandas crosstab function builds a cross-tabulation table that can show the frequency with which certain groups of data appear.
#pandas.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, dropna=True)
#Compute a simple cross-tabulation of two (or more) factors. By default computes a frequency table of the factors unless an array of values and an aggregation function are passed

#Categorical Independent Variable vs Target Variable
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



#It can be inferred that the proportion of male and female applicants is more or less same for both approved and unapproved loans.
Married = pd.crosstab(train['Married'], train['Loan_Status'])
Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Education = pd.crosstab(train['Education'], train['Loan_Status'])
Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar")
plt.show() 
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(4,4)) 
plt.show() 
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(4,4)) 
plt.show()


# Inference
# 1. Proportion of married applicants is higher for approved loans
# 2. Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status

# In[326]:


#Relationship between Categorical Independent Variables and Loan_Status
Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind= "bar", stacked=True, figsize=(4,4))
plt.show()
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind= "bar", stacked=True)
plt.show()


# Inference
# 1. People with credit history as 1 are more likely to get their loans approved.
# 2. Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.

# In[327]:


#Numerical Independent Variable vs Target Variable
# We will try to find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved.

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[328]:


# Here the y-axis represents the mean applicant income. We don’t see any change in the mean income. So, let’s make bins for the applicant income variable based on the values in it and analyze the corresponding loan status for each bin.
bins = [0, 2500, 4000, 6000, 81000]
group =['Low', 'Average', 'High', 'Very high']
train['Income_bin'] = pd.cut(df['ApplicantIncome'], bins, labels=group)

Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind = "bar", stacked=True) 
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')


# It can be inferred that Applicant income does not affect the chances of loan approval which contradicts our hypothesis in which we assumed that if the applicant income is high the chances of loan approval will also be high.

# In[329]:


bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.
# Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.

# In[330]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# We can see that Proportion of loans getting approved for applicants having low Total_Income is very less as compared to that of applicants with Average, High and Very High Income.
# 
# Let’s visualize the Loan amount variable.

# In[331]:


bins = [0,100,200,700]
group = ['Low', 'Average', 'High']
train['LoanAmount_bin'] = pd.cut(df['LoanAmount'], bins, labels=group)
LoanAmount_bin = pd.crosstab(train['LoanAmount_bin'], train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P=plt.ylabel('Percentage')


# It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount as compared to that of High Loan Amount which supports our hypothesis in which we considered that the chances of loan approval will be high when the loan amount is less.

# We will change the 3+ in dependents variable to 3 to make it a numerical variable.We will also convert the target variable’s categories into 0 and 1 so that we can find its correlation with numerical variables. One more reason to do so is few models like logistic regression takes only numeric values as input. We will replace N with 0 and Y with 1.

# In[332]:


train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True) 

train.head()
train.shape


# In[333]:


#Correlation between all numerical variables

matrix = train.corr()
f, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix, vmax=.8, square =True, cmap = "BuPu")


# Most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History-Loan_Status) 
# . LoanAmount is also correlated with CoapplicantIncome.

# In[334]:


# 4. Missing Values
train.isnull().sum()


# In[335]:


# Filling missing values:
# For numerical variables : using mean or median
# For categorical variables : using mode

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

#Now let’s try to find a way to fill the missing values in Loan_Amount_Term. We will look at the value count of the Loan amount term variable.
train['Loan_Amount_Term'].value_counts()

#It can be seen that in loan amount term variable, the value of 360 is repeating the most. So we will replace the missing values in this variable using the mode of this variable.
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

#Numerical value : Loan Amount 

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

#As we can see that all the missing values have been filled in the test dataset. Let’s fill all the missing values in the test dataset too with the same approach.

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[336]:


# 5. Outliers

#As we saw earlier in univariate analysis, LoanAmount contains outliers so we have to treat them as the presence of outliers affects the distribution of the data.
#Having outliers often has a significant effect on the mean and standard deviation and hence affects the distribution. We must take steps to remove outliers from our data sets.
#Due to these outliers bulk of the data in the loan amount is at the left and the right tail is longer. This is called right skewness. One way to remove the skewness is by doing the log transformation. As we take the log transformation, it does not affect the smaller values much, but reduces the larger values. So, we get a distribution similar to normal distribution.

#Let’s visualize the effect of log transformation. We will do the similar changes to the test file simultaneously.

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])


# #6. Evaluation Metrics:
# We can plot the results and compare them with the actual values, i.e. calculate the distance between the predictions and actual values. Lesser this distance more accurate will be the predictions. Since this is a classification problem, we can evaluate our models using any one of the following evaluation metrics:
# 
# 1. Accuracy -
# Using the confusion matrix which is a tabular representation of Actual vs Predicted values. 
# 
# Specificity and Sensitivity plays a crucial role in deriving ROC curve.
# 
# ROC curve
# Receiver Operating Characteristic(ROC) summarizes the model’s performance by evaluating the trade offs between true positive rate (sensitivity) and false positive rate(1- specificity).
# The area under curve (AUC), referred to as index of accuracy(A) or concordance index, is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction power of the model.
# The area of this curve measures the ability of the model to correctly classify true positives and true negatives. We want our model to predict the true classes as true and false classes as false.
# So it can be said that we want the true positive rate to be 1. But we are not concerned with the true positive rate only but the false positive rate too. For example in our problem, we are not only concerned about predicting the Y classes as Y but we also want N classes to be predicted as N.
# 

# In[337]:


# 7. Model Building

# Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.
# Logistic regression is an estimation of Logit function. Logit function is simply a log of odds in favor of the event.
# This function creates a s-shaped curve with the probability estimate, which is very similar to the required step wise function

#dropping Loan_ID variable

# train.head()
# test.head()

train = train.drop('Loan_ID', axis = 1)
test = test.drop('Loan_ID', axis=1)
test.head()


# In[338]:


# X = train.drop(['Loan_Status','Income_bin_Low',
#        'Income_bin_Average', 'Total_Income','Income_bin_High', 'Income_bin_Very high',
#        'Coapplicant_Income_bin_Low', 'Coapplicant_Income_bin_Average',
#        'Coapplicant_Income_bin_High', 'Total_Income_bin_Low',
#        'Total_Income_bin_Average', 'Total_Income_bin_High',
#        'Total_Income_bin_Very high', 'LoanAmount_bin_Low',
#        'LoanAmount_bin_Average', 'LoanAmount_bin_High'],1)
# X = train.drop(['Loan_Status',
#          'Income_bin', 'Coapplicant_Income_bin', 'Total_Income',
#          'Total_Income_bin', 'LoanAmount_bin'], 1)
# # X.columns
# X = train[['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#         'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#         'Loan_Amount_Term', 'Credit_History', 'Property_Area',
#         'LoanAmount_log']].copy()
# test.columns
X = train.drop('Loan_Status', 1)
y = train.Loan_Status
# train.columns
# Now we will make dummy variables for the categorical variables. Dummy variable turns categorical variables into a series of 0 and 1, making them lot easier to quantify and compare. Let us understand the process of dummies first:
# #Consider the “Gender” variable. It has two classes, Male and Female.
# As logistic regression takes only the numerical values as input, we have to change male and female into numerical value.
# Once we apply dummies to this variable, it will convert the “Gender” variable into two variables(Gender_Male and Gender_Female), one for each class, i.e. Male and Female.
# Gender_Male will have a value of 0 if the gender is Female and a value of 1 if the gender is Male.

X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
X.head()

X.shape


# In[339]:


#  We can divide our train dataset into two parts: train and validation. We can train the model on this train part and using that make predictions for the validation part. In this way we can validate our predictions as we have the true predictions for the validation part
# We will use the train_test_split function from sklearn to divide our train dataset. So, first let us import train_test_split.

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

# Here the C parameter represents inverse of regularization strength. Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. Smaller values of C specify stronger regularization. 


# In[340]:


# Let’s predict the Loan_Status for validation set and calculate its accuracy.

pred_cv = model.predict(x_cv)

accuracy_score(y_cv, pred_cv)

# So our predictions are almost 80% accurate, i.e. we have identified 80% of the loan status correctly.


# In[341]:


#Predictions for test data set
# pred_test = model.predict(test)
# test = test.transpose()
# test.head()
# pred_test = model.predict(test)
# x_train.shape
# test.shape
# x_train.head()

test.columns


# In[342]:


pred_test = model.predict(test)


# In[343]:


submission = pd.read_csv('sample_submission_49d68Cx.csv')


# In[344]:


submission['Loan_Status'] = pred_test
test_original = pd.read_csv('tes
                            t_loan.csv')
test_original.head()
submission['Loan_ID']=test_original['Loan_ID']


# In[345]:


#Convert 1 to Y and 0 to N
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace= True)


# In[346]:


submission
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('Loan_Prediction_LogisticRegression.csv')


# In[ ]:


# 8. Robustness  of Model:
# To check the robustness of the model, we use validation. 
# It is a technique which involves reserving a particular sample of a dataset on which you do not train the model. Later, you test your model on this sample before finalizing it.

# Methods of Validation:
# The validation set approach
# k-fold cross validation
# Leave one out cross validation (LOOCV)
# Stratified k-fold cross validation

# Stratified k-fold cross validation :
# Stratification is the process of rearranging the data so as to ensure that each fold is a good representative of the whole.
# For example, in a binary classification problem where each class comprises of 50% of the data, it is best to arrange the data such that in every fold, each class comprises of about half the instances.
# It is generally a better approach when dealing with both bias and variance.
# A randomly selected fold might not adequately represent the minor class, particularly in cases where there is a huge class imbalance.


# In[347]:


#Importing StratifiedKFold
from sklearn.model_selection import StratifiedKFold

#Cross Validation Logistic Model with stratified 5 folds

i=1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
for train_index, test_index in kf.split(X,y):
    print('\n{} of kfold{}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]    
    ytr,yvl = y[train_index],y[test_index]        
    model = LogisticRegression(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)
    pred=model.predict_proba(xvl)[:,1]
    


# In[288]:


#Roc Curve:
get_ipython().run_line_magic('pylab', '')
from sklearn import metrics 
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend(loc=4) 
plt.show()


# In[348]:


# We got an auc value of 0.77
submission['Loan_Status'] = pred_test
submission['Loan_Id'] = test_original['Loan_ID']


# In[349]:


submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('LogisticReg.csv')



# Based on the domain knowledge, we can come up with new features that might affect the target variable. We will create the following three new features:
# 
# Total Income - As discussed during bivariate analysis we will combine the Applicant Income and Coapplicant Income. If the total income is high, chances of loan approval might also be high.
# EMI - EMI is the monthly amount to be paid by the applicant to repay the loan. Idea behind making this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate the EMI by taking the ratio of loan amount with respect to loan amount term.
# Balance Income - This is the income left after the EMI has been paid. Idea behind creating this variable is that if this value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval.
# 

# In[350]:


# test = pd.read_csv('test_loan.csv')

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 
# test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']


# In[351]:


sns.distplot(train['Total_Income']);


# In[352]:


#We can see it is shifted towards left, i.e., the distribution is right skewed. So, let’s take the log transformation to make the distribution normal.
train['Total_Income_log'] = np.log(train['Total_Income']) 
sns.distplot(train['Total_Income_log']); 
# test['Total_Income_log'] = np.log(test['Total_Income'])


# In[353]:


#Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided. Let’s create the EMI feature now.

train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 
# test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


# In[354]:


sns.distplot(train['EMI']);


# In[355]:


#Balance Income Feature
train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal 
# test['Balance Income']=test['Total_Income']-(test['EMI']*1000)


# In[356]:


sns.distplot(train['Balance Income']);


# In[313]:



#Let us now drop the variables which we used to create these new features. Reason for doing this is, the correlation between those old features and these new features will be very high and logistic regression assumes that the variables are not highly correlated. We also wants to remove the noise from the dataset, so removing correlated features will help in reducing the noise too.

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# In[ ]:


# After creating new features, we can continue the model building process. So we will start with logistic regression model and then move over to more complex models like RandomForest and XGBoost.

# We will build the following models in this section.

# Logistic Regression
# Decision Tree
# Random Forest
# XGBoost


# 9. Decision Tree
# Decision tree is a type of supervised learning algorithm(having a pre-defined target variable) that is mostly used in classification problems. In this technique, we split the population or sample into two or more homogeneous sets(or sub-populations) based on most significant splitter / differentiator in input variables.
# 
# Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable.

# In[357]:


from sklearn import tree
i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
for train_index, test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = tree.DecisionTreeClassifier(random_state=1)     
    model.fit(xtr, ytr)   
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)


# In[359]:


#Decision Tree Algorithm : The mean validation accuracy for this model is 0.69
submission['Loan_Status'] = pred_test
submission['Loan_Id'] = test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('DecisionTree_LoanPrediction.csv')

# We got an accuracy of 0.63 which is much lesser than the accuracy from logistic regression model. So let’s build another model, i.e. Random Forest, a tree based ensemble algorithm and try to improve our model by improving the accuracy.


# 10. Random Forest
# RandomForest is a tree based bootstrapping algorithm wherein a certain no. of weak learners (decision trees) are combined to make a powerful prediction model.
# For every individual learner, a random sample of rows and a few randomly chosen variables are used to build a decision tree model.
# Final prediction can be a function of all the predictions made by the individual learners.
# In case of regression problem, the final prediction can be mean of all the predictions.

# In[361]:


from sklearn.ensemble import RandomForestClassifier
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(random_state=1, max_depth=10)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 

    pred_test = model.predict(test)


# The mean validation accuracy for this model is 0.766
# 
# We will try to improve the accuracy by tuning the hyperparameters for this model. We will use grid search to get the optimized values of hyper parameters. Grid-search is a way to select the best of a family of hyper parameters, parametrized by a grid of parameters.
# 
# We will tune the max_depth and n_estimators parameters. max_depth decides the maximum depth of the tree and n_estimators decides the number of trees that will be used in random forest model.

# In[364]:


from sklearn.model_selection import GridSearchCV
# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)


# In[365]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3, random_state=1)
#fit grid search model

grid_search.fit(x_train, y_train)


# In[366]:


# Estimating the optimized value 
grid_search.best_estimator_


# In[368]:


# So, the optimized value for the max_depth variable is 3 and for n_estimator is 41. Now let’s build the model using these optimized values.

i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred2=model.predict_proba(test)[:,1]


# In[369]:


submission['Loan_Status']=pred_test            # filling Loan_Status with predictions submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y 
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
# Converting submission file to .csv format 
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('RandomForest_LoanPrediction.csv')
#We got an accuracy of 0.7638 from the random forest model on leaderboard.


# In[370]:


# Let us find the feature importance now, i.e. which features are most important for this problem. We will use feature_importances_ attribute of sklearn to do so.

importances=pd.Series(model.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8))


# We can see that Credit_History is the most important feature followed by Balance Income, Total Income, EMI. So, feature engineering helped us in predicting our target variable.

# 11. XGBoost
# XGBoost works only with numeric variables and we have already replaced the categorical variables with numeric variables. Let’s have a look at the parameters that we are going to use in our model.
# 
# n_estimator: This specifies the number of trees for the model.
# max_depth: We can specify maximum depth of a tree using this parameter.

# In[373]:


from xgboost import XGBClassifier
i = 1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = XGBClassifier(n_estimators=50, max_depth=4)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred3=model.predict_proba(test)[:,1]


# In[374]:


# The mean validation accuracy for this model is 0.79
submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('XGBoost_LoanPrediction.csv')

# We got an accuracy of 0.73611 with this model.

# After trying and testing 4 different algorithms, the best accuracy is achieved by Logistic Regression (0.7847), followed by RandomForest (0.7638).

