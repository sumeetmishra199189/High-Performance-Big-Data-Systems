
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import forest
import numpy as np
import time
from sklearn.metrics import f1_score


# In[6]:


airlineData = pd.read_csv("2008.csv.bz2")
pd.set_option('display.max_columns', 500)


# In[12]:


airlineData.columns


# In[22]:


airlineData['Dest'].nunique()


# In[44]:


#airlineData['ArrTime'].isna().sum()/airlineData['DepTime'].shape[0]


# In[26]:


airlineData['ArrDelay'].quantile(0.25)


# In[47]:


airlineData.head()


# In[50]:


airlineData = airlineData.drop(['Origin','Dest','TailNum','UniqueCarrier','FlightNum','Cancelled','Diverted','CancellationCode', 'CarrierDelay','WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'],axis = 1)


# In[51]:


airlineData.shape


# In[53]:


airlineData.dropna().shape[0]/airlineData.shape[0]


# In[54]:


airlineData = airlineData.dropna()


# In[57]:


airlineData.head()


# In[65]:


airlineData['DepTimeInMins'] = airlineData['DepTime'].apply(lambda x : (int(x/100)*60 + x%100))
airlineData['CRSDepTimeInMins'] = airlineData['CRSDepTime'].apply(lambda x : (int(x/100)*60 + x%100))


# In[66]:


airlineData['ArrTimeInMins'] = airlineData['ArrTime'].apply(lambda x : (int(x/100)*60 + x%100))
airlineData['CRSArrTimeInMins'] = airlineData['CRSArrTime'].apply(lambda x : (int(x/100)*60 + x%100))


# In[70]:


airlineData['IsDelayed'] = airlineData['ArrDelay'] > 18
airlineData['IsDelayed'] = airlineData['IsDelayed'].map(int)


# In[76]:


airlineData['IsDelayed'].value_counts()


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(airlineData.loc[:, ~airlineData.columns.isin(['IsDelayed'])], airlineData['IsDelayed'], test_size=0.25, random_state=42)


# In[107]:


airlineData.to_csv("AirlineReduced",index = False)


# In[2]:


print(pd.read_csv("AirlineReduced").columns)


#
#
## In[86]:
#
#
#clf = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)
#clf.fit(X_test, y_test)
#
#
## In[89]:
#
#
#y_pred = clf.predict(X_train)
#
#
## In[91]:
#
#
#roc_auc_score(y_train, y_pred)
#
#
## In[102]:
#
#
#myForest = forest.forest(maxDepth=5,numTrees= 10,verbose = True)
#TrainX,TrainY = np.array(X_test),np.array(y_test)
#Xtest,yTest = np.array(X_test),np.array(y_test)
#start = time.time()
#myForest.trainForest(TrainX,TrainY)
#
#finalPredictions = myForest.predict(Xtest)
#
#print("Time elapsed is ", time.time()-start)
#print("Accuracy is: " ,sum(finalPredictions==yTest)/len(yTest))
#
#
## In[104]:
#
#
#f1_score(yTest,finalPredictions)

