#!/usr/bin/env python
# coding: utf-8

# ## TASK 2 : TO EXPLORE SUPERVISED MACHINE LEARNING

# In[2]:


## IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


## IMPORTING THE DATA

df=pd.read_csv('book.csv')
print('DATA IS IMPORTED SUCCESSFULLY')
df


# In[4]:


## FIRST FIVE ROWS OF THE DATA

df.head()


# In[5]:


## DATA DESCRIPTION

df.describe()


# In[6]:


## CHECKING FOR THE MISSING VALUE

df.isnull().sum()


# #### HENCE NO VALUE IS MISSING IN THE DATA

# In[58]:


## SCATTER PLOT OF DATA

plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours studied vs Marks scored', fontsize=20)
plt.scatter(df.Hours,df.scores)
plt.show()


# #### PREPARING DATA

# In[75]:


## DEFINING OF VARIABLES

X = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[76]:


## TRAIN AND TEST SET

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# ### TRAINING MODEL

# In[77]:


from sklearn.linear_model import LinearRegression
linreg= LinearRegression()
linreg.fit(X_train,y_train)


# In[78]:


##SLOPE(A0) AND INTERCEPT(A1)

print("A0 =" , linreg.intercept_,"A1 =",linreg.coef_)


# In[79]:


##PLOTTING REGRESSION LINE ON TRAIN SET


yO = linreg.intercept_ + linreg.coef_*X_train
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression plot for train set',fontsize=10)
plt.scatter(X_train,y_train,color='black',marker='*')
plt.plot(X_train,yO,color='blue')
plt.show()


# In[80]:


## PREDICTINF SCORES FOR TEST DATA

prediction = linreg.predict(X_test)
print(prediction)


# In[81]:


## PLOTTING REGRESSION LINE FOR TEST SET

plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression plot for test data')
plt.plot(X_test,prediction,color='green')
plt.scatter(X_test,y_test,color='orange',marker='*')
plt.show()


# In[85]:


## COMPARING ACTUAL AND PREDICTED VALUE


dataset=pd.DataFrame({'ACTUAL VALUE':y_test,'PREDICTED VALUE':prediction})
dataset


# #### ACCURACY OF MODEL

# In[86]:


from sklearn import metrics
metrics.r2_score(y_test,prediction)


# #### ACCURACY IS APPROXIMATELY 94% WHICH SHOWS THAT THIS MODEL IS A GOOD MODEL

# In[87]:


## PREDICTING ERROR

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Absolute Error :', metrics.mean_absolute_error(y_test,prediction))
print('Mean Squared Error :', metrics.mean_squared_error(y_test,prediction))


# In[88]:


## SCORE PREDICTION

final_pred=linreg.predict([[9.25]])
print('After studying for 9.25 hours the student is expected to score ',final_pred)

