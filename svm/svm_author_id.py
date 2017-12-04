#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/home/nan/ud120-projects/tools")
from email_preprocess import preprocess
from class_vis import prettyPicture, output_image

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(features_train.shape)
print(features_test.shape)
import time
from sklearn import svm

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
# clf = svm.SVC(kernel="rbf", C=10000)
clf = svm.SVC(kernel="linear")
start_time = time.time()
clf.fit(features_train, labels_train)
print("---Training took %s seconds ---" % (time.time() - start_time))
pred=clf.predict(features_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print("Accuracy: ", accuracy)

print(pred[10])
print(pred[26])
print(pred[50])
print(sum(pred))


# prettyPicture(clf, features_test, labels_test)


#########################################################
### your code goes here ###

#########################################################


