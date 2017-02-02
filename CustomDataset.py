import pandas as pd
import numpy as np

class CustomDataset:
    def __init__(self, train_df, test_df, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.trainMatrix = self._create_train_matrix(train_df)
        self.testRatings, self.testNegatives = self._create_test_data(test_df, train_df)
    
    def _create_train_matrix(self, train_df):
        train_dict = {(int(row['UserId']), int(row['BGGId'])): 1 for _, row in train_df.iterrows()}
        class TrainMatrix:
            def __init__(self, data, num_users, num_items):
                self.data = data
                self.shape = (num_users, num_items)
            
            def keys(self):
                return self.data.keys()
            
            def __getitem__(self, key):
                return self.data.get(key, 0)

            def __contains__(self, key):
                return key in self.data  # Explicitly check if key is in dictionary
        
        return TrainMatrix(train_dict, self.num_users, self.num_items)
    
    def _create_test_data(self, test_df, train_df):
        test_ratings = test_df[['UserId', 'BGGId', 'Rating']].values.tolist()
        test_negatives = []
        train_dict = {(int(row['UserId']), int(row['BGGId'])): 1 for _, row in train_df.iterrows()}
        
        for u in test_df['UserId'].unique():
            user_negatives = []
            rated_items = set(train_df[train_df['UserId'] == u]['BGGId'].values)
            rated_items.update(test_df[test_df['UserId'] == u]['BGGId'].values)
            for _ in range(100):
                j = np.random.randint(self.num_items)
                while j in rated_items:
                    j = np.random.randint(self.num_items)
                user_negatives.append(j)
            test_negatives.append(user_negatives)
        
        return test_ratings, test_negatives
