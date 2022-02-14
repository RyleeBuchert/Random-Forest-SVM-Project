from decision_tree import DecisionTree
import pandas as pd
import numpy as np
import random

class RandomForest:

    # class constructor
    def __init__(self):
        self.forest = []

    # method to train random forest model
    def grow_forest(self,      
            X = None, # X_train
            Y = None, # Y_train
            num_trees = None, # Number of trees in forest
            num_features = None # Number of features trained on
            ):
        
        # get X and Y dataframes
        self.X = X
        self.Y = Y
        self.Y_name = Y.name

        # get feature info
        self.features = self.X.columns.tolist()
        self.num_features = len(self.features)
        
        # get dataset info
        self.concat_data = pd.concat([self.Y, self.X], axis=1)
        self.sample_size = len(self.concat_data)
        self.columns = self.concat_data.columns.tolist()

        # get hyparameters
        self.forest_size = num_trees if num_trees else 100
        self.feature_subset = num_features if num_features else int(np.sqrt(self.num_features))

        # create decision trees and grow forest
        for i in range(self.forest_size):
            tree_sample = self.bootstrap_sample()
            X_train = tree_sample.drop(columns=self.Y_name, axis=1)
            Y_train = tree_sample[self.Y_name]
            tree = DecisionTree()
            tree.build_tree(X_train, Y_train, min_samples=4, max_depth=3)
            self.forest.append(tree)

    # get bootstrap aggregation sample
    def bootstrap_sample(self):
        data_list = []
        indexes = self.X.index.tolist()
        for i in range(self.sample_size):
            temp_list = []
            idx = random.choice(indexes)
            row = self.concat_data.iloc[[idx]]
            for j in self.columns:
                temp_list.append(row.loc[idx][j])
            data_list.append(temp_list)
        data_sample = pd.DataFrame(data_list, columns=self.columns)
        col_subset = random.sample(self.features, self.feature_subset)
        col_subset.insert(0, self.Y_name)
        return data_sample[col_subset]

    # predict random forest accuracy (need to finish)
    def predict(self, X_test, Y_test):
        # test_concat_data = pd.concat([Y_test, X_test], axis=1)
        # for idx in test_concat_data.index.tolist():
        #     row = test_concat_data.iloc[[idx]]
        #     for tree in self.forest:

        forest_predictions = []
        for tree in self.forest:
            forest_predictions.append(tree.predict(X_test, Y_test)[0])

        forest_predictions_df = pd.DataFrame(forest_predictions)
        for pred in forest_predictions:
            print()





if __name__ == "__main__":

    # import train and test data
    golf_data = pd.read_csv('data\\golf_data2.csv')
    X_train = golf_data.drop(columns='PlayGolf', axis=1)
    Y_train = golf_data['PlayGolf']

    golf_test = pd.read_csv('data\\golf_data2_test.csv')
    X_test = golf_test.drop(columns='PlayGolf', axis=1)
    Y_test = golf_test['PlayGolf']

    # train random forest and predict results
    RF = RandomForest()
    RF.grow_forest(X_train, Y_train, num_trees=10)
    print(RF.predict(X_test, Y_test))