import pandas as pd
import numpy as np

class Node:

    # class constructor
    def __init__(
            self,
            X = None, # X_train
            Y = None, # Y_train
            is_root = None, # Boolean for root node
            is_leaf = None, # Boolean for leaf node
            parent = None, # Pointer to parent node
            feature_cat = None, # Feature category of node
            depth = None, # Depth of current node
            max_depth = None, # Max depth of tree
            min_samples = None, # Min samples to split on
            split = None, # Split method (Entropy, GINI, etc.)
            ):

        # get X and Y dataframes
        self.X = X
        self.Y = Y

        # get node information
        self.is_root = is_root if is_root else False
        self.is_leaf = is_leaf if is_leaf else False
        self.parent = parent if parent else None

        # get feature information
        self.feature_category = feature_cat if feature_cat else None
        self.best_feature = None
        self.children = []

        # stopping point/split information
        self.depth = depth if depth else 0
        self.max_depth = max_depth
        self.min_split_samples = min_samples
        self.split_method = split

        # get all classes
        self.classes = np.unique(self.Y)
        self.num_classes = len(self.classes)
        self.Y_name = self.Y.name

        # get all features
        self.features = self.X.columns.tolist()
        self.num_features = len(self.features)

        # get percent of each class in dataset
        self.concat_data = pd.concat([self.Y, self.X], axis=1)
        self.class_count_dictionary = {i: {} for i in self.classes}
        for i in self.classes:
            self.class_count_dictionary[i].update({'Count': len(self.concat_data.loc[self.concat_data[self.Y_name] == i])})
            self.class_count_dictionary[i].update({'Percent': (self.class_count_dictionary[i]['Count'] / len(self.concat_data))})

    # method to find the best feature using cross-entropy loss
    def pick_attribute(self):

        # split data into categorical and continuous dataframes
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        continuous_data = self.concat_data.select_dtypes(include=numerics)
        continuous_columns = continuous_data.columns.tolist()
        continuous_data = pd.concat([self.Y, continuous_data], axis=1)
        
        categorical_data = self.concat_data.drop(columns=continuous_columns, axis=1)
        categorical_columns = categorical_data.columns.tolist()
        categorical_columns.remove(self.Y_name)

        # get information gain for each continuous feature
        feature_results = {}
        self.feature_splits = {}
        for i in continuous_columns:
            # get sorted set of all values in continuous column
            sorted_vals_set = set(sorted(continuous_data[i]))
            total_count = len(self.concat_data)

            continuous_split_results = {}
            for j in sorted_vals_set:
                # split data into subsets for each value split
                remainder = 0
                smaller_data = continuous_data[continuous_data[i] <= j]
                larger_data = continuous_data[continuous_data[i] > j]

                # calculate cardinality of split data
                smaller_cardinality = (len(smaller_data) / total_count)
                larger_cardinality = (len(larger_data) / total_count)

                # get class probability list
                smaller_class_probs = []
                larger_class_probs = []
                for k in self.classes:
                    smaller_count = len(smaller_data.loc[smaller_data[self.Y_name] == k])
                    if len(smaller_data) > 0:
                        smaller_class_probs.append(smaller_count / len(smaller_data))
                    else:
                        smaller_class_probs.append(0)

                    larger_count = len(larger_data.loc[larger_data[self.Y_name] == k])
                    if len(larger_data) > 0:
                        larger_class_probs.append(larger_count / len(larger_data))
                    else:
                        larger_class_probs.append(0)

                # calculate remainder
                if self.split_method == 'Cross-Entropy':
                    remainder += (smaller_cardinality * self.cross_entropy(smaller_class_probs)) + (larger_cardinality * self.cross_entropy(larger_class_probs))
                elif self.split_method == 'GINI':
                    remainder += (smaller_cardinality * self.GINI_index(smaller_class_probs)) + (larger_cardinality * self.GINI_index(larger_class_probs))
                elif self.split_method == 'Misclassification':
                    remainder += (smaller_cardinality * self.misclassification(smaller_class_probs)) + (larger_cardinality * self.misclassification(larger_class_probs))
                
                continuous_split_results.update({j: remainder})
            
            # get best continuous split
            best_split_value = min(continuous_split_results, key=continuous_split_results.get)
            self.feature_splits.update({i: best_split_value})

            # get information gain for best continuous value
            h_prior_list = []
            for k in self.classes:
                h_prior_list.append(self.class_count_dictionary[k]['Percent'])
            
            if self.split_method == 'Cross-Entropy':
                h_prior = self.cross_entropy(h_prior_list)
            elif self.split_method == 'GINI':
                h_prior = self.GINI_index(h_prior_list)
            elif self.split_method == 'Misclassification':
                h_prior = self.misclassification(h_prior_list)
            
            information_gain = h_prior - continuous_split_results[best_split_value]
            feature_results.update({i: information_gain})

        # get information gain for each categorical feature
        for i in categorical_columns:
            # get categories for each feature and number of instances
            feature_categories = np.unique(self.X[i])
            total_count = len(self.concat_data)

            remainder = 0
            subset_dictionary = {}
            categorical_count_dict = {}
            for j in feature_categories:
                # split data into subsets for each feature category
                subset_dictionary.update({j: self.concat_data.loc[self.concat_data[i] == j]})
                categorical_count_dict.update({j: {}})
                
                # calculate cardinality
                categorical_count_dict[j].update({'Total': len(subset_dictionary[j])})
                category_cardinality = (categorical_count_dict[j]['Total'] / total_count)

                # get class probability list
                class_probs = []
                for k in self.classes:
                    categorical_count_dict[j].update({k: len(subset_dictionary[j].loc[subset_dictionary[j][self.Y_name] == k])})
                    class_probs.append(categorical_count_dict[j][k] / categorical_count_dict[j]['Total'])

                # calculate remainder
                if self.split_method == 'Cross-Entropy':
                    remainder += category_cardinality * self.cross_entropy(class_probs)
                elif self.split_method == 'GINI':
                    remainder += category_cardinality * self.GINI_index(class_probs)
                elif self.split_method == 'Misclassification':
                    remainder += category_cardinality * self.misclassification(class_probs)
            
            # get information gain for each attribute
            h_prior_list = []
            for k in self.classes:
                h_prior_list.append(self.class_count_dictionary[k]['Percent'])
            
            if self.split_method == 'Cross-Entropy':
                h_prior = self.cross_entropy(h_prior_list)
            elif self.split_method == 'GINI':
                h_prior = self.GINI_index(h_prior_list)
            elif self.split_method == 'Misclassification':
                h_prior = self.misclassification(h_prior_list)
            
            information_gain = h_prior - remainder
            feature_results.update({i: information_gain})
        
        # return feature with the highest information gain
        self.best_feature = max(feature_results, key=feature_results.get)

    # method to calculate cross-entropy
    def cross_entropy(self, class_probs):
        remainder = 0
        for q in class_probs:
            if q == 0 or q == 1:
                remainder += 0
            else:
                remainder += (-1 * q) * np.log2(q)
        return remainder

    # method to calculate GINI index
    def GINI_index(self, class_probs):
        remainder = 0
        for q in class_probs:
            if q == 0 or q == 1:
                remainder += 0
            else:
                remainder += q * (1 - q)
        return remainder

    # method to calculate misclassification rate
    def misclassification(self, class_probs):
        return 1 - max(class_probs)

    # method to build a decision tree with X/Y training sets
    def grow_tree(self):
        
        # check stop conditions and grow tree
        if (self.depth < self.max_depth) and (len(self.concat_data) >= self.min_split_samples):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            self.pick_attribute()

            # create nodes if continuous best feature
            if self.concat_data[self.best_feature].dtypes in numerics:
                new_nodes_dict = {}
                feature_subsets_dict = {
                    'Smaller': self.concat_data[self.concat_data[self.best_feature] <= self.feature_splits[self.best_feature]],
                    'Larger': self.concat_data[self.concat_data[self.best_feature] > self.feature_splits[self.best_feature]]
                }
                for i in feature_subsets_dict:
                    # if only one class value remains, max depth is reached, or feature subset too small, create a leaf node
                    if (len(np.unique(feature_subsets_dict[i][self.Y_name]))==1) or (self.depth==self.max_depth) or (len(feature_subsets_dict[i])<self.min_split_samples):
                        new_nodes_dict.update({i: Node(
                            X = feature_subsets_dict[i].drop(columns=self.Y_name),
                            Y = feature_subsets_dict[i][self.Y_name],
                            is_leaf = True,
                            parent = self,
                            feature_cat = (i, self.feature_splits[self.best_feature]),
                            depth = self.depth + 1,
                            max_depth = self.max_depth,
                            min_samples = self.min_split_samples,
                            split = self.split_method
                        )})
                        self.children.append(new_nodes_dict[i])

                    # else, create a new decision node and continue recursive grow
                    else:
                        new_nodes_dict.update({i: Node(
                            X = feature_subsets_dict[i].drop(columns=self.Y_name),
                            Y = feature_subsets_dict[i][self.Y_name],
                            parent = self,
                            feature_cat = (i, self.feature_splits[self.best_feature]),
                            depth = self.depth + 1,
                            max_depth = self.max_depth,
                            min_samples = self.min_split_samples,
                            split = self.split_method
                        )})
                        self.children.append(new_nodes_dict[i])
                        new_nodes_dict[i].grow_tree()
            
            # create nodes if categorical best feature
            else:
                best_feature_categories = np.unique(self.X[self.best_feature])
                new_nodes_dict = {}
                feature_subsets_dict = {}
                for i in best_feature_categories:
                    feature_subsets_dict.update({i: self.concat_data.loc[self.concat_data[self.best_feature] == i]})

                    # if only one class value remains, max depth is reached, or feature subset too small, create a leaf node
                    if (len(np.unique(feature_subsets_dict[i][self.Y_name]))==1) or (self.depth==self.max_depth) or (len(feature_subsets_dict[i])<self.min_split_samples):
                        new_nodes_dict.update({i: Node(
                            X = feature_subsets_dict[i].drop(columns=self.Y_name),
                            Y = feature_subsets_dict[i][self.Y_name],
                            is_leaf = True,
                            parent = self,
                            feature_cat = i,
                            depth = self.depth + 1,
                            max_depth = self.max_depth,
                            min_samples = self.min_split_samples,
                            split = self.split_method
                            )})
                        self.children.append(new_nodes_dict[i])                    
                    
                    # else, create a new decision node and continue recursive grow
                    else:
                        new_nodes_dict.update({i: Node(
                            X = feature_subsets_dict[i].drop(columns=self.Y_name),
                            Y = feature_subsets_dict[i][self.Y_name],
                            parent = self,
                            feature_cat = i,
                            depth = self.depth + 1,
                            max_depth = self.max_depth,
                            min_samples = self.min_split_samples,
                            split = self.split_method
                            )})
                        self.children.append(new_nodes_dict[i])  
                        new_nodes_dict[i].grow_tree()


class DecisionTree:

    # class constructor
    def __init__(self):
        self.root = None
    
    # method to build tree
    def build_tree(self, X, Y, max_depth=None, min_samples=None, split_method=None): # add hyperparameters
        self.max_depth = max_depth if max_depth else 10
        self.min_split_samples = min_samples if min_samples else 20
        self.split_method = split_method if split_method else 'Cross-Entropy'
        if self.root is None:
            self.root = Node(is_root='True',
                            X = X,
                            Y = Y,
                            max_depth = self.max_depth,
                            min_samples = self.min_split_samples,
                            split = self.split_method)
        self.root.grow_tree()


if __name__ == "__main__":

    golf_data = pd.read_csv('data\\golf_data2.csv')
    X_train = golf_data.drop(columns='PlayGolf', axis=1)
    Y_train = golf_data['PlayGolf']    

    DT = DecisionTree()
    DT.build_tree(X_train, Y_train, max_depth=5, min_samples=4, split_method='Cross-Entropy')
