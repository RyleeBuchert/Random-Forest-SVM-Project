from sklearn.model_selection import train_test_split
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
            num_features = None, # Number of features trained on
            min_samples = None, # Min split samples for trees
            max_depth = None, # Max depth for decision trees
            split_method = None # Split method for trees
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
        self.min_split_samples = min_samples if min_samples else 10
        self.max_tree_depth = max_depth if max_depth else 5
        self.split_method = split_method if split_method else 'Cross-Entropy'

        # create decision trees and grow forest
        for i in range(self.forest_size):
            tree_sample = self.bootstrap_sample()
            X_train = tree_sample.drop(columns=self.Y_name, axis=1)
            Y_train = tree_sample[self.Y_name]
            tree = DecisionTree()
            tree.build_tree(X = X_train,
                            Y = Y_train,
                            min_samples = self.min_split_samples,
                            max_depth = self.max_tree_depth,
                            split_method = self.split_method
                            )
            self.forest.append(tree)
            print(i)

    # get bootstrap aggregation sample
    def bootstrap_sample(self):
        data_list = []
        indexes = self.X.index.tolist()
        col_subset = random.sample(self.features, self.feature_subset)
        col_subset.insert(0, self.Y_name)
        temp_df = self.concat_data[col_subset] 
        for i in range(self.sample_size):    
            data_list.append(temp_df.loc[random.choice(indexes), :].values.flatten().tolist())
        return pd.DataFrame(data_list, columns=col_subset)

    # predict random forest accuracy (need to finish)
    def predict(self, X_test, Y_test):
        # iterate over trees in forest and generate predictions
        forest_predictions = []
        for tree in self.forest:
            forest_predictions.append(tree.predict(X_test, Y_test)[0])
        
        # remove invalid predictions and create df
        for pred in forest_predictions:
            if len(pred) < len(Y_test):
                forest_predictions.remove(pred)
        forest_predictions_df = pd.DataFrame(forest_predictions)

        # determine final predictions based on majority vote from forest
        prediction_list = []
        for col in forest_predictions_df.columns.tolist():
            vote = forest_predictions_df[col].mode()
            prediction_list.append(vote[0])

        # count correct answers and return model accuracy
        total_correct = 0
        total_number = len(Y_test)
        for idx, prediction in enumerate(prediction_list):
            if prediction == Y_test.iloc[idx]:
                total_correct += 1
        return (prediction_list, (total_correct/total_number))


if __name__ == "__main__":

    # # import train and test data
    # golf_data = pd.read_csv('data\\golf_data.csv')
    # X_train = golf_data.drop(columns='PlayGolf', axis=1)
    # Y_train = golf_data['PlayGolf']

    # golf_test = pd.read_csv('data\\golf_data_test.csv')
    # X_test = golf_test.drop(columns='PlayGolf', axis=1)
    # Y_test = golf_test['PlayGolf']

    # # import spirals data
    # spirals = pd.read_csv('data\\spirals.csv')
    # spirals_X = spirals.drop(columns='class', axis=1)
    # spirals_Y = spirals['class']
    # X_train, X_test, Y_train, Y_test = train_test_split(spirals_X, spirals_Y, test_size = 0.2)

    # # import blobs data
    # blobs = pd.read_csv('data\\blobs.csv')
    # blobs_X = blobs.drop(columns='class', axis=1)
    # blobs_Y = blobs['class']
    # X_train, X_test, Y_train, Y_test = train_test_split(blobs_X, blobs_Y, test_size = 0.2)

    # import digit data
    digit = pd.read_csv('data\\digit_bins.csv')
    digit_X = digit.drop(columns='label', axis=1)
    digit_Y = digit['label']
    X_train, X_test, Y_train, Y_test = train_test_split(digit_X, digit_Y, test_size = 0.2)

    # drop indexes
    X_train.reset_index(drop=True, inplace=True)
    Y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    Y_test.reset_index(drop=True, inplace=True)

    # train random forest and predict results
    RF = RandomForest()
    RF.grow_forest(X_train, Y_train, num_trees=50, min_samples=10, max_depth=5)
    print(RF.predict(X_test, Y_test)[1])