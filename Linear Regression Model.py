
# coding: utf-8

# In[1]:


import gc
import numpy as np
import pandas as pd
from dfply import *
import random
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE,r2_score,mean_absolute_error,explained_variance_score


# In[2]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.width',180)

# Import Functions
import sys
sys.path.append('E:\Python work')
import Py_Function as fs


# In[3]:


#   Read Files
Train = pd.read_csv('D:/Case Study/train.csv')


# In[4]:


Train.head()


# In[5]:


# Check the count of Missing value in data
# From below output no missing values in data
Train.apply(lambda x : sum(x.isnull()),axis = 0)


# In[6]:


# Check the summary of the variables
Train.describe()


# In[7]:


# For checking outlires in data
Q1 = Train.quantile(0.25)
Q3 = Train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[8]:


# If output is false then there is no missing values in data
print(((Train < (Q1 - 1.5 * IQR)) |(Train > (Q3 + 1.5 * IQR))))


# In[9]:


# Remove the outlires
#Train = Train[~((Train < (Q1 - 1.5 * IQR)) |(Train > (Q3 + 1.5 * IQR))).any(axis=1)]

# Decision Tree is good dealing with outlire so outlire does not removed
Train.shape


# In[10]:


# Correlation of all Predictor variable with Target Variable 
# Instead of scatter plot correlation is best option to check the relation between dependent and independent variable 
Train.corr()[['critical_temp']]


# In[11]:


# Another way to check relation between predictor and target variable is scatter plot
plt.scatter(Train.wtd_std_ThermalConductivity,Train.critical_temp)
# correlation between Critical_temp and wtd_std_ThermalConductivity is 0.721271 but scatter plot does not show good relation 
# correlation is best option to select relation between dependent variable


# In[12]:


# for checking Mullticollinearity of data correlation matrix
Train.corr()
# From below correlation matrix we can see there is lot of varibles has correlation is high
# i.e chnce of Mullticollinearity in data


# In[13]:


# Split data in train and test set
Y = Train['critical_temp']
X = Train.drop(columns=['critical_temp'])
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=100)


# In[14]:


# some variables in above correlation matrix is show higher correlation 
# so best way to identify multicollinearity is compute VIF
# For each X, calculate VIF and save in dataframe
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif
# From below table VIF > 10 from all variables so it is high multicollinearity problem


# In[15]:


# For ploting the data take sample from data frame
Trans1 = Train.sample(frac =.20)
sns.distplot(Trans1['critical_temp'],bins=20)
# from the below plot critical temp distribution does not look like normal


# In[16]:


# check the predictor variables distribution look like normal curve or not
# The best way to the distribution of variables is qqplot or histogram or normality test
sns.distplot(Trans1['entropy_fie'],bins=10)
qqplot(Trans1.entropy_fie, line='s')
# from below plot the entropy_fie variable is follow normal distribution


# In[17]:


# normality test of Response variable
from scipy.stats import shapiro
stat, p = shapiro(Train.critical_temp)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Normal(Gaussian) (fail to reject H0)')
else:
    print('Sample does not look Normal(Gaussian) (reject H0)')


# In[ ]:


# The response variable is non normal and also there is high multicollinearity in the data
# So instead of linear regression model non parametric model will be good fit for this data 
# Because non parametric models has does not any assumptions like DecisionTreeRegressor


# In[18]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# In[20]:


# Maximum Tree Depth: This is the maximum number of nodes from the root node of the tree. 
# Once a maximum depth of the tree is met, we must stop splitting adding new nodes. 
# Deeper trees are more complex and are more likely to overfit the training data.

# Min_samples_leaf = The minimum number of samples required to be at a leaf node

dt = DecisionTreeRegressor(max_depth=8,
                           #max_features = 50,
                           min_samples_leaf =0.02,
                           random_state=4)


# In[21]:


# Fit dt to the training set
Model1 = dt.fit(x_train, y_train)


# In[22]:


MSE_CV = - cross_val_score(Model1,x_train,y_train,cv = 15,
                        scoring = 'neg_mean_squared_error',
                        n_jobs = -1)


# In[23]:


y_pred_train = Model1.predict(x_train)
y_pred_test = Model1.predict(x_test)


# In[24]:


# Compute mse_dt
mse_dt = MSE(y_train, y_pred_train)

# Coeff of determination
R = r2_score(y_train, y_pred_train)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_train,y_pred_train)

# mean_absolute_error
MAE = mean_absolute_error(y_train, y_pred_train)

# Print rmse_dt
print("Train set MAE of dt: {:.2f}".format(MAE))
print("Train set MSE of dt: {:.2f}".format(mse_dt))
print("Train set RMSE of dt: {:.2f}".format(rmse_dt))
print("Train set coefficient of determination: {:.2f}".format(R))
print("Train set Explained_variance_score: {:.2f}".format(EVS))


# In[25]:


# Compute mse_dt
mse_dt = MSE(y_test, y_pred_test)

# Coeff of determination
R = r2_score(y_test, y_pred_test)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_test,y_pred_test)

# mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)

# Print rmse_dt
print("Test set MAE of dt: {:.2f}".format(MAE))
print("Test set MSE of dt: {:.2f}".format(mse_dt))
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
print("Test set coefficient of determination: {:.2f}".format(R))
print("Test set Explained_variance_score: {:.2f}".format(EVS))


# In[26]:


# Compute mse_dt of CV
print('CV MSE:{:.2f}'.format(MSE_CV.mean())) 

# Compute rmse_dt
rmse_CV = MSE_CV.mean()**(1/2)
print('CV RMSE:{:.2f}'.format(rmse_CV))


# In[27]:


sns.distplot(y_test,bins=20)
sns.distplot(y_pred_test,bins=20)


# In[28]:


# The RMSE of Training and Testing data is nearly equal ~ 16
# But for better results we want to check important variables from model 
# and built model on important variables


# In[29]:


Residual = y_test-y_pred_test
# normality test of Response variable
from scipy.stats import shapiro
stat, p = shapiro(Residual)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Normal(Gaussian) (fail to reject H0)')
else:
    print('Sample does not look Normal(Gaussian) (reject H0)')

sns.distplot(Residual,bins=20)
# From the Ressidual plot distribution of error looking like log-normal distribution


# In[30]:


Model1.feature_importances_


# In[31]:


Train.shape


# In[35]:


A = Train.columns[[2,3,4,5,8,9,10,12,13,14,17,18,22,27,31,33,31,39,42,48,51,55,61,64,67,70,72,74,76]]


# In[36]:


x_train = x_train[A]
x_test = x_test[A]


# In[37]:


dt = DecisionTreeRegressor(max_depth=8,
                           #max_features = 50,
                           #min_samples_leaf = 2,
                           min_samples_leaf =0.02,
                           random_state=4)


# In[38]:


# Fit dt to the training set on importatnt variable
Model2 = dt.fit(x_train, y_train)


# In[39]:


# Compute y_pred
y_pred_train = Model2.predict(x_train)
y_pred_test = Model2.predict(x_test)


# In[40]:


# Compute mse_dt
mse_dt = MSE(y_train, y_pred_train)

# Coeff of determination
R = r2_score(y_train, y_pred_train)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_train,y_pred_train)

# mean_absolute_error
MAE = mean_absolute_error(y_train, y_pred_train)

# Print rmse_dt
print("Train set MAE of dt: {:.2f}".format(MAE))
print("Train set MSE of dt: {:.2f}".format(mse_dt))
print("Train set RMSE of dt: {:.2f}".format(rmse_dt))
print("Train set coefficient of determination: {:.2f}".format(R))
print("Train set Explained_variance_score: {:.2f}".format(EVS))


# In[41]:


# Compute mse_dt
mse_dt = MSE(y_test, y_pred_test)

# Coeff of determination
R = r2_score(y_test, y_pred_test)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_test,y_pred_test)

# mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)

# Print rmse_dt
print("Test set MAE of dt: {:.2f}".format(MAE))
print("Test set MSE of dt: {:.2f}".format(mse_dt))
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
print("Test set coefficient of determination: {:.2f}".format(R))
print("Test set Explained_variance_score: {:.2f}".format(EVS))


# In[42]:


# From the all metrics of Model1 and Model2 
# RMSE of model1 is less than model2
# coefficient of determination R^2 of model1 and model2 is same
# I.e All variables are important in the data
# i.e Variable has no main effect with critical temp but may be intraction effect present
# So model 2 has best fit


# In[43]:


sns.distplot(y_test,bins=20)
sns.distplot(y_pred_test,bins=20)


# In[ ]:


# save the model for future use using 
from sklearn.externals import joblib
filename = 'finalized_model.sav'
joblib.dump(Model1, filename)


# In[ ]:


# Use the model on data 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)


# In[44]:


# For predicting the critical temp using the Support vector machine Regression Model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# In[50]:


# Split data in train and test set
Y = Train['critical_temp']
X = Train.drop(columns=['critical_temp'])
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=100)


# In[51]:


# The shape of distribution curve of target variable critical_temp is looks like RBF model
# So we select here RBF kernel
seed = 100

# Parameters for tuning
parameters = [{'kernel': ['rbf'], 
               'gamma': [1e-8],
               'C': [1000]}]

# I have choose very value of gamma because if gamma increases then model overfit 

Model_SVM_RBF = GridSearchCV(SVR(epsilon = 0.01), parameters, cv = 5)
Model_SVM_RBF.fit(x_train, y_train)


# In[52]:


y_pred_train = Model_SVM_RBF.predict(x_train)
y_pred_test = Model_SVM_RBF.predict(x_test)


# In[53]:


# Compute mse_dt
mse_dt = MSE(y_train, y_pred_train)

# Coeff of determination
R = r2_score(y_train, y_pred_train)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_train,y_pred_train)

# mean_absolute_error
MAE = mean_absolute_error(y_train, y_pred_train)

# Print rmse_dt
print("Train set MAE of dt: {:.2f}".format(MAE))
print("Train set MSE of dt: {:.2f}".format(mse_dt))
print("Train set RMSE of dt: {:.2f}".format(rmse_dt))
print("Train set coefficient of determination: {:.2f}".format(R))
print("Train set Explained_variance_score: {:.2f}".format(EVS))


# In[54]:


# Compute mse_dt
mse_dt = MSE(y_test, y_pred_test)

# Coeff of determination
R = r2_score(y_test, y_pred_test)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_test,y_pred_test)

# mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)

# Print rmse_dt
print("Test set MAE of dt: {:.2f}".format(MAE))
print("Test set MSE of dt: {:.2f}".format(mse_dt))
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
print("Test set coefficient of determination: {:.2f}".format(R))
print("Test set Explained_variance_score: {:.2f}".format(EVS))


# In[ ]:


# Both Train and test data has same RMSE so model is good


# In[55]:


sns.distplot(y_test,bins=20)
sns.distplot(y_pred_test,bins=20)


# In[56]:


# For predicting the critical temp using the BayesianRidge Regression machine learning Model
# Because Ridge Regression working good if large values of VIF (high multicollinearity)
# Ridge regression are powerful techniques generally used for models in presence of a ‘large’ number of features.
from sklearn.linear_model import BayesianRidge


# In[57]:


seed = 10
clf = BayesianRidge(alpha_1=1e-01, alpha_2=1e-02,
                    lambda_1=1e-01, lambda_2=1e-02,
                    n_iter=200,tol=0.001)
clf.fit(x_train, y_train)


# In[58]:


y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)


# In[59]:


# Compute mse_dt
mse_dt = MSE(y_train, y_pred_train)

# Coeff of determination
R = r2_score(y_train, y_pred_train)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_train,y_pred_train)

# mean_absolute_error
MAE = mean_absolute_error(y_train, y_pred_train)

# Print rmse_dt
print("Train set MAE of dt: {:.2f}".format(MAE))
print("Train set MSE of dt: {:.2f}".format(mse_dt))
print("Train set RMSE of dt: {:.2f}".format(rmse_dt))
print("Train set coefficient of determination: {:.2f}".format(R))
print("Train set Explained_variance_score: {:.2f}".format(EVS))


# In[60]:


# Compute mse_dt
mse_dt = MSE(y_test, y_pred_test)

# Coeff of determination
R = r2_score(y_test, y_pred_test)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# explained_variance_score
EVS = explained_variance_score(y_test,y_pred_test)

# mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)

# Print rmse_dt
print("Test set MAE of dt: {:.2f}".format(MAE))
print("Test set MSE of dt: {:.2f}".format(mse_dt))
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
print("Test set coefficient of determination: {:.2f}".format(R))
print("Test set Explained_variance_score: {:.2f}".format(EVS))


# In[ ]:


# Both Train and Test data has same RMSE


# In[61]:


sns.distplot(y_test,bins=20)
sns.distplot(y_pred_test,bins=20)

