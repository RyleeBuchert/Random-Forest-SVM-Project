from sklearn.model_selection import train_test_split
from random_forest import RandomForest
import pandas as pd

if __name__ == "__main__":

    # log_file = open('test_data\\spirals_results.txt', 'w')

    # # import spirals data
    # spirals = pd.read_csv('data\\spirals.csv')
    # spirals_X = spirals.drop(columns='class', axis=1)
    # spirals_Y = spirals['class']

    # # train random forest and predict results
    # forest_sizes = [50, 100, 150]
    # split_methods = ['Misclassification', 'GINI', 'Cross-Entropy']
    # feature_subset_sizes = [2]
    # for forest_size in forest_sizes:
    #     for split in split_methods:
    #         for subset_size in feature_subset_sizes:
    #             X_train, X_test, Y_train, Y_test = train_test_split(spirals_X, spirals_Y, test_size = 0.2)

    #             X_train.reset_index(drop=True, inplace=True)
    #             Y_train.reset_index(drop=True, inplace=True)
    #             X_test.reset_index(drop=True, inplace=True)
    #             Y_test.reset_index(drop=True, inplace=True)

    #             RF = RandomForest()
    #             RF.grow_forest(X_train, Y_train, num_trees=forest_size, min_samples=10, max_depth=5, split_method=split, feature_subset_size=subset_size)
    #             results = RF.predict(X_test, Y_test)[1]
                
    #             log_file.write('Spirals Data: Forest Size = '+str(forest_size)+', Split = '+split+', Subset Size = '+str(subset_size)+', Model Accuracy = '+str(results) + '\n')

    # log_file.close()




    # adult_log_file = open('test_data\\adult_results.txt', 'w')

    # # import adult data
    # adult = pd.read_csv('data\\adult_data.csv')
    # adult_X = adult.drop(columns='income', axis=1)
    # adult_Y = adult['income']

    # # train random forest and predict results
    # forest_sizes = [25, 50, 100]
    # split_methods = ['Misclassification', 'GINI', 'Cross-Entropy']
    # feature_subset_sizes = [2, 4]
    # for forest_size in forest_sizes:
    #     for split in split_methods:
    #         for subset_size in feature_subset_sizes:
    #             X_train, X_test, Y_train, Y_test = train_test_split(adult_X, adult_Y, test_size = 0.2)

    #             X_train.reset_index(drop=True, inplace=True)
    #             Y_train.reset_index(drop=True, inplace=True)
    #             X_test.reset_index(drop=True, inplace=True)
    #             Y_test.reset_index(drop=True, inplace=True)

    #             RF = RandomForest()
    #             RF.grow_forest(X_train, Y_train, num_trees=forest_size, min_samples=10, max_depth=5, split_method=split, feature_subset_size=subset_size)
    #             results = RF.predict(X_test, Y_test)[1]
                
    #             adult_log_file.write('Adult Data: Forest Size = '+str(forest_size)+', Split = '+split+', Subset Size = '+str(subset_size)+', Model Accuracy = '+str(results) + '\n')

    # adult_log_file.close()




    # digit_log_file = open('test_data\\digit_results2.txt', 'w')

    # # import adult data
    # digit = pd.read_csv('data\\digit_bins.csv')
    # digit_X = digit.drop(columns='label', axis=1)
    # digit_Y = digit['label']

    # # train random forest and predict results
    # forest_sizes = [50]
    # split_methods = ['Misclassification', 'GINI', 'Cross-Entropy']
    # feature_subset_sizes = [5, 10, 25]
    # for forest_size in forest_sizes:
    #     for split in split_methods:
    #         for subset_size in feature_subset_sizes:
    #             X_train, X_test, Y_train, Y_test = train_test_split(digit_X, digit_Y, test_size = 0.2)

    #             X_train.reset_index(drop=True, inplace=True)
    #             Y_train.reset_index(drop=True, inplace=True)
    #             X_test.reset_index(drop=True, inplace=True)
    #             Y_test.reset_index(drop=True, inplace=True)

    #             RF = RandomForest()
    #             RF.grow_forest(X_train, Y_train, num_trees=forest_size, min_samples=10, max_depth=5, split_method=split, feature_subset_size=subset_size)
    #             results = RF.predict(X_test, Y_test)[1]
                
    #             digit_log_file.write('Digit Data: Forest Size = '+str(forest_size)+', Split = '+split+', Subset Size = '+str(subset_size)+', Model Accuracy = '+str(results) + '\n')

    # digit_log_file.close()




    # import blobs data
    blobs = pd.read_csv('data\\blobs3.csv')
    blobs_X = blobs.drop(columns='class', axis=1)
    blobs_Y = blobs['class']

    blobs3_file = open('test_data\\blobs3_results.txt', 'w')

    # train random forest and predict results
    forest_sizes = [50, 100, 150]
    split_methods = ['Misclassification', 'GINI', 'Cross-Entropy']
    feature_subset_sizes = [2]
    for forest_size in forest_sizes:
        for split in split_methods:
            for subset_size in feature_subset_sizes:
                X_train, X_test, Y_train, Y_test = train_test_split(blobs_X, blobs_Y, test_size = 0.2)

                X_train.reset_index(drop=True, inplace=True)
                Y_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                Y_test.reset_index(drop=True, inplace=True)

                RF = RandomForest()
                RF.grow_forest(X_train, Y_train, num_trees=forest_size, min_samples=10, max_depth=5, split_method=split, feature_subset_size=subset_size)
                results = RF.predict(X_test, Y_test)[1]
                
                blobs3_file.write('Blobs3 Data: Forest Size = '+str(forest_size)+', Split = '+split+', Subset Size = '+str(subset_size)+', Model Accuracy = '+str(results) + '\n')

    blobs3_file.close()



    # import blobs data
    blobs = pd.read_csv('data\\blobs4.csv')
    blobs_X = blobs.drop(columns='class', axis=1)
    blobs_Y = blobs['class']

    blobs4_file = open('test_data\\blobs4_results.txt', 'w')

    # train random forest and predict results
    forest_sizes = [25, 50, 75]
    split_methods = ['Misclassification', 'GINI', 'Cross-Entropy']
    feature_subset_sizes = [2]
    for forest_size in forest_sizes:
        for split in split_methods:
            for subset_size in feature_subset_sizes:
                X_train, X_test, Y_train, Y_test = train_test_split(blobs_X, blobs_Y, test_size = 0.2)

                X_train.reset_index(drop=True, inplace=True)
                Y_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                Y_test.reset_index(drop=True, inplace=True)

                RF = RandomForest()
                RF.grow_forest(X_train, Y_train, num_trees=forest_size, min_samples=10, max_depth=5, split_method=split, feature_subset_size=subset_size)
                results = RF.predict(X_test, Y_test)[1]
                
                blobs4_file.write('Blobs4 Data: Forest Size = '+str(forest_size)+', Split = '+split+', Subset Size = '+str(subset_size)+', Model Accuracy = '+str(results) + '\n')

    blobs4_file.close()



    # import blobs data
    blobs = pd.read_csv('data\\blobs5.csv')
    blobs_X = blobs.drop(columns='class', axis=1)
    blobs_Y = blobs['class']

    blobs5_file = open('test_data\\blobs5_results.txt', 'w')

    # train random forest and predict results
    forest_sizes = [25, 50, 75]
    split_methods = ['Misclassification', 'GINI', 'Cross-Entropy']
    feature_subset_sizes = [2]
    for forest_size in forest_sizes:
        for split in split_methods:
            for subset_size in feature_subset_sizes:
                X_train, X_test, Y_train, Y_test = train_test_split(blobs_X, blobs_Y, test_size = 0.2)

                X_train.reset_index(drop=True, inplace=True)
                Y_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                Y_test.reset_index(drop=True, inplace=True)

                RF = RandomForest()
                RF.grow_forest(X_train, Y_train, num_trees=forest_size, min_samples=10, max_depth=5, split_method=split, feature_subset_size=subset_size)
                results = RF.predict(X_test, Y_test)[1]
                
                blobs5_file.write('Blobs5 Data: Forest Size = '+str(forest_size)+', Split = '+split+', Subset Size = '+str(subset_size)+', Model Accuracy = '+str(results) + '\n')

    blobs5_file.close()