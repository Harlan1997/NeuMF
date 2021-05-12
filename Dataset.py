from datetime import time
import os
import numpy as np
from numpy.lib.arraysetops import unique 
import pandas as pd
from pandas.core.frame import DataFrame
import copy
import datetime
import csv
import random
import sklearn
import sklearn.model_selection


_original_data_path = 'data\\ml-1m\\ratings.dat'

class Dataset(object):
    def __init__ (self, num_test_negatives = 100, num_train_negatives = 4):
        self.num_test_negatives = num_test_negatives
        self.num_train_negatives = num_train_negatives
        self.num_users, self.num_items, self.train_positves, self.train_negatives, self.test_positives, self.test_negatives, self.validation_data = self.load_data(_original_data_path)
        self.X_train, self.Y_train, self.X_val, self.Y_val = self.get_train_instances(self.train_positves, self.train_negatives)
        self.X_test, self.Y_test = self.get_test_instances(self.test_positives, self.test_negatives)
        
    def load_data(self, file_path):
        print("loading file into dataframe...")
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(file_path, index_col=False, sep='::', names=names, engine='python')
        
        #reindex with base 0
        original_items = df['item_id'].unique()
        original_users = df['user_id'].unique()
        num_users = len(original_users)
        num_items = len(original_items)
        user_map = {user: idx for idx, user in enumerate(original_users)}
        item_map = {item: idx for idx, item in enumerate(original_items)}
        print("Reindex dataframe...")
        df['item_id'] = df['item_id'].apply(lambda item:item_map[item])
        df['user_id'] = df['user_id'].apply(lambda user:user_map[user])
        rating_dict = {}

        print("Store data into dictionary...")
        for row in df.itertuples():
            user_id = getattr(row, 'user_id')
            item_id = getattr(row, 'item_id')
            rating = getattr(row, 'rating')
            timestamp = getattr(row, 'timestamp')
            if user_id not in rating_dict:
                rating_dict[user_id] = []
            rating_dict[user_id].append((item_id, rating, timestamp))

        print("Sorting data according to timestamp...")
        for user_id in rating_dict:
            rating_dict[user_id] = sorted(rating_dict[user_id],key=lambda x:(x[2]),reverse=True) 
        print("get train data and test data...")
        test_positives = []
        train_positives = []
        test_negatives = []
        train_negatives = []
        validation_data = []
        all_items = set(range(num_items))
        for user_id in rating_dict:
            rated_items = set(rating_dict[user_id])
            all_negatives = all_items.difference(rated_items)
            latest_record = rating_dict[user_id].pop(0)
            timestamp = latest_record[2]
            item_id = latest_record[0]
            rating = latest_record[1]
            test_positives.append((user_id, item_id, rating, timestamp))
            for record in rating_dict[user_id]:
                timestamp = record[2]
                item_id = record[0]
                rating = record[1]
                train_positives.append((user_id, item_id, rating, timestamp))
                sample_items = random.sample(all_negatives, self.num_train_negatives)
                train_negatives.append(sample_items)
            sample_items = random.sample(all_negatives, self.num_test_negatives)
            test_negatives.append(sample_items)
        assert len(train_positives) == len(train_negatives)
        print("writing data to csv...")
        self.write_data_to_csv(train_positives, train_negatives, test_positives, test_negatives)

        return num_users, num_items, train_positives, train_negatives, test_positives, test_negatives, validation_data

    def write_data_to_csv(self, train_positives, train_negatives, test_positives, test_negatives):
        with open('data\\ml-1m\\train_positives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for tup in train_positives:
                user_id = tup[0]
                item_id = tup[1]
                rating = tup[2]
                timestamp = tup[3]
                writer.writerow([user_id, item_id, rating, timestamp])
        with open('data\\ml-1m\\test_positives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for tup in test_positives:
                user_id = tup[0]
                item_id = tup[1]
                rating = tup[2]
                timestamp = tup[3]
                writer.writerow([user_id, item_id, rating, timestamp])

        with open('data\\ml-1m\\train_negatives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(train_negatives)):
                writer.writerow(train_negatives[i])
                
        with open('data\\ml-1m\\test_negatives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(test_negatives)):
                writer.writerow(test_negatives[i])
        
    def get_train_instances(self, train_positives, train_negatives):
        user_input, item_input, labels = [], [], []
        for i in range(len(train_positives)):
            record = train_positives[i]
            user = record[0]
            item = record[1]
            user_input.append(user)
            item_input.append(item)
            labels.append(1)
            for item in train_negatives[i]:
                user_input.append(user)
                item_input.append(item)
                labels.append(0)
        user_input = np.array(user_input)
        item_input = np.array(item_input)
        labels = np.array(labels)
        np.random.seed(200)
        np.random.shuffle(user_input)
        np.random.seed(200)
        np.random.shuffle(item_input)
        np.random.seed(200)
        np.random.shuffle(labels)
        user_input_train, user_input_val, labels_train, labels_val = sklearn.model_selection.train_test_split(user_input, labels, test_size=0.1, random_state=0)
        item_input_train, item_input_val, labels_train, labels_val = sklearn.model_selection.train_test_split(item_input, labels, test_size=0.1, random_state=0)
        X_train = [user_input_train, item_input_train]
        Y_train = labels_train
        X_val = [user_input_val, item_input_val]
        Y_val = labels_val 
        return X_train, Y_train, X_val, Y_val

    def get_test_instances(self, test_positives, test_negatives):
        user_input, item_input, labels = [], [], []
        for i in range(len(test_positives)):
            record = test_positives[i]
            user = record[0]
            item = record[1]
            user_input.append(user)
            item_input.append(item)
            labels.append(1)
            for item in test_negatives[i]:
                user_input.append(user)
                item_input.append(item)
                labels.append(0)
        np.random.seed(200)
        np.random.shuffle(user_input)
        np.random.seed(200)
        np.random.shuffle(item_input)
        np.random.seed(200)
        np.random.shuffle(labels)
        X_test = [np.array(user_input), np.array(item_input)]
        Y_test = np.array(labels)
        return X_test, Y_test        

if __name__ == '__main__':
    dataset = Dataset()   
    print(dataset.num_items)