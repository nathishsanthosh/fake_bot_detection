
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

filepath = 'D:/data/'

bots = pd.read_csv("d:/data/bots_data.csv",encoding='ISO-8859-1')
nonbots = pd.read_csv("d:/data/nonbots_data.csv",encoding="ISO-8859-1" )

#Creating Bots identifying condition
#bots[bots.listedcount>10000]
condition = (bots.screen_name.str.contains("bot", case=False)==True)|(bots.description.str.contains("bot", case=False)==True)|(bots.location.isnull())|(bots.verified==False)

bots['screen_name_binary'] = (bots.screen_name.str.contains("bot", case=False)==True)
bots['description_binary'] = (bots.description.str.contains("bot", case=False)==True)
bots['location_binary'] = (bots.location.isnull())
bots['verified_binary'] = (bots.verified==False)
print("Bots shape: {0}".format(bots.shape))

#Creating NonBots identifying condition
condition = (nonbots.screen_name.str.contains("bot", case=False)==False)| (nonbots.description.str.contains("bot", case=False)==False) |(nonbots.location.isnull()==False)|(nonbots.verified==True)

nonbots['screen_name_binary'] = (nonbots.screen_name.str.contains("bot", case=False)==False)
nonbots['description_binary'] = (nonbots.description.str.contains("bot", case=False)==False)
nonbots['location_binary'] = (nonbots.location.isnull()==False)
nonbots['verified_binary'] = (nonbots.verified==True)
print("Nonbots shape: {0}".format(nonbots.shape))

#Joining Bots and NonBots dataframes
df = pd.concat([bots, nonbots])
print("DataFrames created...")

#Splitting data randombly into train_df and test_df
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2)
print("Randomly splitting the dataset into training and test, and training classifiers...\n")




#Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(min_samples_split=50, min_samples_leaf=200)

#80%
X_train = train_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']] #train_data
y_train = train_df['bot'] #train_target

#20%
X_test = test_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']] #test_Data
y_test = test_df['bot'] #test_target

#Training on decision tree classifier
model = clf.fit(X_train, y_train)

#Predicting on test data
predicted = model.predict(X_test)

#Checking accuracy
print("Random Forest Classifier Accuracy: {0}".format(accuracy_score(y_test, predicted)))



#80%
X_train = train_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']] #train_data
y_train = train_df['bot'] #train_target

#20%
X_test = test_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']] #test_Data
y_test = test_df['bot'] #test_target

#Training on SVM Classifier
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(X_train, y_train)
predicted=linear_svc.predict(X_test)
print("\nSVM with Linear Kernal\n")
print("--------------------------------\n")
Classification = 'SVM using Linear Kernal'

#Checking accuracy
print("SVM using Linear Kernal: {0}".format(accuracy_score(y_test, predicted)))



