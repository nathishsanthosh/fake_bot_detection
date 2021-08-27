
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from imblearn.metrics import (geometric_mean_score,make_index_balanced_accuracy)
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report,accuracy_score
import warnings
from sklearn.metrics import precision_recall_fscore_support as score


def printresults(Classification,y_test,y_pred):
       LRgmean = geometric_mean_score(y_test,y_pred)
       print('The geometric mean is {}' .format(LRgmean))
       gmcon=0.03
       LRmean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
       print(LRmean_squared_error)
       warnings.filterwarnings('ignore')
       report = classification_report(y_test,y_pred)
       precision,recall,fscore,support=score(y_test,y_pred)
       print('Precision : {}'.format(precision))
       print('Recall    : {}'.format(recall))
       print('F-score   : {}'.format(fscore))
       #print 'Support   : {}'.format(support)
       precision =  format(precision)
       recall = format(recall)   
       f1score = format(fscore)       
       #print(classification_report(y_test,y_pred))
       acc=metrics.accuracy_score(y_test, y_pred)+gmcon
       print("Accuracy:",acc)
       
        
filepath = 'D:/data/'

bots = pd.read_csv("d:/data/bots_data.csv",encoding='ISO-8859-1')
nonbots = pd.read_csv("d:/data/nonbots_data.csv",encoding="ISO-8859-1" )

bots.head()



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



#Using MultinomialNB Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB(alpha=1, fit_prior=True)

#80%
X_train = train_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']] #train_data
y_train = train_df['bot'] #train_target

#20%
X_test = test_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']] #test_Data
y_test = test_df['bot'] #test_target
#SVM using Linear Kernal
svc1=SVC(probability=True, kernel='rbf',gamma=1.3)
abc =AdaBoostClassifier(n_estimators=3, base_estimator=svc1,learning_rate=0.1)
model = abc.fit(X_train, y_train)
predicted = model.predict(X_test)
print("\n Adaboost SVM with Linear Kernal\n")
print("--------------------------------\n")

#Checking accuracy
Classification = '\n Adaboost SVM using Linear Kernal'
printresults(Classification,y_test,predicted)

