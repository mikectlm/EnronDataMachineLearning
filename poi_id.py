#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import train_test_split
import sklearn.metrics as skm
from sklearn.model_selection import KFold

best_params = []

def clf_tester(clf, features, labels, folds = 50):
    print clf
    kf = KFold(n_splits=folds)
        
    accuracy = []
    precision = []
    recall = []    
    for train_idx, test_idx in kf.split(features): 
        
        
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(skm.accuracy_score(labels_test, predictions))
        precision.append(skm.precision_score(labels_test, predictions))
        recall.append(skm.recall_score(labels_test, predictions))

        #best_params.append(clf.best_params_)
    print "Accuracy:  {}".format(np.mean(accuracy))
    print "Precision: {}".format(np.mean(precision))
    print "Recall:    {}".format(np.mean(recall))
    print ''
    return np.mean(precision), np.mean(recall)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# setting some useful lists
data_set = data_dict
finance_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_list = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
all_features = finance_list + email_list
email_list.insert(0, 'poi')

features_list = ['poi']

#create data set using all features
data = featureFormat(data_set, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

## DATA EXPLORATION
# General Information
print "Number of Data Points: " + str(len(data_dict))
poi_count = 0
for poi in data_dict:
    if data_dict[poi]["poi"] == 1: 
        poi_count += 1        
print "Number of POI's: " + str(poi_count)
print "Number of Non-POI's: " + str(len(data_dict) - poi_count)
print "Total Number of Features: " + str(len(data_dict.items()[0][1].keys()))
print ''    

# Discovering the counts of NaN for each features
null_count = { x:0 for i,x in enumerate(all_features)}
for feature in features:
    for i, item in enumerate(feature):
        if item == 0:
            null_count[all_features[i+1]] += 1

for nulls in null_count:    
    print "{0:>40}  {1:>4} {2:>6.2f}".format(nulls, str(null_count[nulls]), float(null_count[nulls])/float(len(data_dict)) )

print ''    

# Discovering the counts of NaN for each Data Point
null_counts_user = { x:0. for i,x in enumerate(data_set.keys())}

for pers in data_set:
    count = 0.
    for feature in data_set[pers]:        
        if data_set[pers][feature] == 'NaN':
            count += 1.
        null_counts_user[pers] = count  / len(data_set[pers])

# Print only the data points with above 80% NaN        
for nulls in null_counts_user:    
    if null_counts_user[nulls] > 0.8:
        print "{0:>40}  {1:>4}".format(nulls, str(null_counts_user[nulls]))

### Task 2: Remove outliers

#Cleaning Records
data_dict.pop('TOTAL')
data_dict.pop('LOCKHART EUGENE E')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
print ''

data = featureFormat(data_set, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


features_train, features_test, poi_train, poi_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#scaler = preprocessing.MinMaxScaler()
#features_scaled = scaler.fit_transform(features)

## Select 10 best using SelectKBest
print 'Selected 10-Best '
kbest = SelectKBest(k=10)
kbest.fit(features_train, poi_train)
kbestlist = []
for i, k in enumerate(kbest.get_support(indices=True)):
    features_list.append(all_features[k+1])
    kbestlist.append([all_features[k+1], kbest.pvalues_[i], kbest.scores_[i]])    
print ''

kbestlist.sort(key=lambda tup: tup[2], reverse=True)


for ll in kbestlist:
    print "{0:>25}  {1:.5f} {2:>8.5f}".format(ll[0], ll[1], ll[2])

### Task 3: Create new feature(s)
### Adding Features

features_list.append('from_poi_ratio') 
features_list.append('to_poi_ratio')
features_list.append('total_exp_wealth')

for pers in data_set:
    if math.isnan(float(data_set[pers]['from_poi_to_this_person'])) or math.isnan(float(data_set[pers]['to_messages'])):
        to_ratio_feature = 0
    else:
        to_ratio_feature = float(data_set[pers]['from_poi_to_this_person'])/float(data_set[pers]['to_messages'])
    
    if math.isnan(float(data_set[pers]['from_this_person_to_poi'])) or math.isnan(float(data_set[pers]['from_messages'])):
        from_ratio_feature = 0
    else:
        from_ratio_feature = float(data_set[pers]['from_this_person_to_poi'])/float(data_set[pers]['from_messages'])
        
    if math.isnan(float(data_set[pers]['total_payments'])):
        tp = 0
    else:
        tp = float(data_set[pers]['total_payments'])
    
    if math.isnan(float(data_set[pers]['total_stock_value'])):
        tsv = 0
    else:
        tsv = float(data_set[pers]['total_stock_value'])
  
    fin_feature = tp + tsv
    
    data_set[pers]['from_poi_ratio'] = to_ratio_feature
    data_set[pers]['to_poi_ratio'] = from_ratio_feature
    
    data_set[pers]['total_exp_wealth'] = fin_feature


all_features.append('from_poi_ratio') 
all_features.append('to_poi_ratio')
all_features.append('total_exp_wealth')

#All Features
#features_list = ['poi', 'salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_poi_ratio', 'total_exp_wealth']

# Fewer Features
features_list = ['poi', 'salary', 'loan_advances', 'bonus', 'deferred_income', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_poi_ratio', 'total_exp_wealth']
    
### Store to my_dataset for easy export below.
my_dataset = data_set

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaled on Features
scaler = preprocessing.MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# scale
features_train, features_test, poi_train, poi_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

print 'SVC using GridSearchCV'
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[i for i in range(1,20)], 'tol': [0.0001, 0.0002, 0.0003,0.005]}
svr = SVC()
gs_svr = GridSearchCV(svr, parameters)
clf_tester(gs_svr, features_scaled, labels, 5)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf_tester(gnb, features_scaled, labels, 5)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, n_init = 200, max_iter = 600, tol=0.00005)
clf_tester(kmeans, features_scaled, labels, 5)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 60, learning_rate  = 1.25)
clf_tester(clf, features_scaled, labels, 5)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)