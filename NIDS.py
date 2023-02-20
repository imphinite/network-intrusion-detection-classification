from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from math import ceil, log2
import numpy as np
import threading 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics # is used to create classification results
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # is used for plotting the decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pickle
from utils import NIDS_utils as NIDSUtils

# If you want to use Intel Extension for Scikit-learn* uncomment the following line
# from sklearnex import patch_sklearn
# patch_sklearn()


def classifier_model():
    # EXAMPLE OF A CLASSIFIER MODEL INPUT YOUR CLASSIFIER MODEL HERE
    ############################### YOUR CODE HERE ###############################



    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)





    return clf



def save_model(clf, filename):
    print('Saving Model: ' + filename + '...')
    pickle.dump(clf, open(filename, 'wb'))
    print('Model saved')


def main():

    target_label = 'attack_cat'

    # For Label
    if target_label == "Label":
        feature_cols = ['sport', 'dsport', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
        'sloss', 'Sload', 'Dload', 'Dpkts', 'smeansz', 'dmeansz', 'Sjit',
        'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'ct_state_ttl',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_dst_src_ltm']


    # For attack_cat
    if target_label == "attack_cat":
        feature_cols = ['dsport', 'dur', 'sbytes', 'sttl', 'Dload', 'smeansz', 'dmeansz',
        'ct_state_ttl', 'ct_srv_dst']
    
    data_frame = NIDSUtils.read_data('UNSW-NB15-BALANCED-TRAIN.csv')

    data_frame = NIDSUtils.preprocess_data(data_frame)

    data_frame = NIDSUtils.randomize_data(data_frame)

    train, test, validate = NIDSUtils.split_data(data_frame)

    del data_frame


    X_train, y_train = NIDSUtils.set_X_y(train, feature_cols, target_label)

    X_test, y_test = NIDSUtils.set_X_y(test, feature_cols, target_label)



    # Scale data
    print('Scaling data...')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('Data scaled')



    # Create and train model
    print('Creating and training model...')
    clf = classifier_model()
    clf.fit(X_train, y_train)
    print('Model created and trained')

    # Make predictions
    print('Making predictions...')
    y_pred = clf.predict(X_test)
    print('Predictions made')

    # Evaluate model
    print('Evaluating model...')
    NIDSUtils.metrics_report(clf,y_test, y_pred)
    print('Model evaluated')

    # Save Model 
    save_model(clf, 'model.sav')



if __name__ == '__main__':
    main()



