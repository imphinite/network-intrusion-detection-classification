import pandas as pd
from math import ceil, log2
import numpy as np

from sklearn import metrics # is used to create classification results

import warnings
warnings.filterwarnings("ignore")


default_header = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
       'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload',
       'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
       'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime',
       'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
       'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
       'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat',
       'Label']

def preprocess_data(data_frame):
        # ct_src_ltm has a space in the column name
    data_frame = data_frame.rename(columns={'ct_src_ ltm': 'ct_src_ltm'})

    # if column has an empty value, add it to the list of empty columns
    print('Finding empty columns...')
    empty_columns = []
    for column in data_frame.columns:
        if data_frame[column].isnull().any() or (data_frame[column].astype(str).str.strip() == '').any():
            empty_columns.append(column)

    print('Empty columns: ', empty_columns)

    # Relace "Backdoors" with "Backdoor" in attack_cat column
    print('Replacing "Backdoors" with "Backdoor" in attack_cat column...')
    data_frame['attack_cat'] = data_frame['attack_cat'].replace('Backdoors', 'Backdoor')
    print('"Backdoors" replaced with "Backdoor" in attack_cat column')

    # Convert hex values in sport and dsport columns to decimal
    print('Converting hex values in sport and dsport columns to decimal...')
    # convert '-' to 0
    data_frame['sport'] = data_frame['sport'].replace('-', 0)
    data_frame['dsport'] = data_frame['dsport'].replace('-', 0)
    data_frame['sport'] = data_frame['sport'].apply(lambda x: int(str(x), 16))
    data_frame['dsport'] = data_frame['dsport'].apply(lambda x: int(str(x), 16))
    print('Hex values in sport and dsport columns converted to decimal')
    # print(data_frame['sport'].head())


    # Replce empty values with 0
    print('Replacing empty values with 0...')
    known_empty_columns = ['ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'attack_cat']

    for column in empty_columns:
        if column == 'attack_cat':
            data_frame[column] = data_frame[column].fillna('None')
            data_frame[column] = data_frame[column].replace(' ', 'None')
        elif column in known_empty_columns:
            data_frame[column] = data_frame[column].fillna(0)
            data_frame[column] = data_frame[column].replace(' ', 0)


    print("Empty values replaced with 0")

    # Covert ct_ftp_cmd column to int
    print('Converting ct_ftp_cmd to int...')
    data_frame['ct_ftp_cmd'] = data_frame['ct_ftp_cmd'].astype(int)
    print('ct_ftp_cmd converted to int')

    # Remove all rows with empty values
    print("Ensuring no leading or trailing spaces...")
    data_frame = data_frame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    print('Facorizing columns...')
    to_fact = ["proto", "state", "service", "attack_cat"]
    facto_ref = {}

    for col in to_fact:
        # print(col, ":", training_data[col].unique())
        temp_array = data_frame[col].unique()
        temp_array.sort()
        temp_dict = {}
        for i, item in enumerate(temp_array):
            if item not in temp_dict:
                temp_dict[item] = i
        facto_ref[col] = temp_dict

    for col in to_fact:
        print(col, ":", facto_ref[col])
        data_frame[col] = data_frame[col].map(facto_ref[col])
    print('Columns factored')


    # converting columns to their appropriate data types
    print('Converting columns to appropriate data types...')
    data_frame['is_ftp_login'] = data_frame['is_ftp_login'].astype(bool)
    data_frame['is_sm_ips_ports'] = data_frame['is_sm_ips_ports'].astype(bool)
    data_frame['Label'] = data_frame['Label'].astype(bool)
    data_frame['ct_flw_http_mthd'] = data_frame['ct_flw_http_mthd'].astype(int)

    # remove alll rows with missing values
    print('Removing all rows with missing values...')
    data_frame = data_frame.dropna()
    print('All rows with missing values removed')


    int_columns = ['proto', 'state', 'sbytes',
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
        'dmeansz', 'trans_depth', 'res_bdy_len', 'Stime',
        'Ltime','ct_state_ttl', 'ct_flw_http_mthd',
        'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat']

    # Finding optimal data type for each column
    print('Finding optimal data type for each integer column...')
    for col in int_columns:
        min_val = data_frame[col].min()
        max_val = data_frame[col].max()

        range_val = max_val - min_val 

        bits_required = ceil(log2(range_val))

        if bits_required < 8:
            int_type = np.int8
        elif bits_required < 16:
            int_type = np.int16
        elif bits_required < 32:
            int_type = np.int32
        else:
            int_type = np.int64

        data_frame[col] = data_frame[col].astype(int_type) 

    print("Data types converted to optimal data types...")

    
    return data_frame


def read_data(csv_file):
    print('Reading data...')
    data_frame = pd.read_csv(csv_file, dtype={'srcip':str, 'sport':str, 'dstip':str, 'dsport':str, 'proto':str, 'state':str, 'service':str,'ct_ftp_cmd':str, 'attack_cat':str})
    # if data does not have a header

    print('Data read')
    return data_frame

def randomize_data(data_frame, rand_state=42):
    print('Randomizing data...')
    data_frame = data_frame.sample(frac=1, random_state=rand_state)
    print('Data randomized')
    return data_frame

def split_data(data_frame):
    print('Splitting into training(70%), validation(15%), and test sets(15%)...')
    train, validate, test = np.split(data_frame, [int(.7*len(data_frame)), int(.85*len(data_frame))])
    print('Splitting complete')
    return train, validate, test



def set_X_y(data_frame, feature_cols,target_label):
    X = data_frame.drop(['srcip','dstip','attack_cat','Label'], axis=1)
    X = X[feature_cols]
    y = data_frame[target_label]
    return X, y


def metrics_report(clf,y_test, y_pred):
    # print classfier estimators
    # if hasattr(clf, 'estimators_'):
    #     print("Classfier Estimators: " + str(clf.estimators_))
    print("Classfier Name: " + clf.__class__.__name__)
    print(metrics.classification_report(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted'))
    # print("Recall:",metrics.recall_score(y_test, y_pred, average='weighted'))
    # print("F1 Score:",metrics.f1_score(y_test, y_pred, average='weighted'))
    # print("ROC AUC Score:",metrics.roc_auc_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

