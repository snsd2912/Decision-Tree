from huytv.tree_gini import evaluate_algorithm, decision_tree, decision_tree_lib, str_column_to_float, load_csv
import numpy as np


filename = 'data_banknote_authentication.csv'
schema = ["variance", "skewness", "curtosis", "entropy"]
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print("Shape of data:", np.array(dataset).shape)
# evaluate algorithm
n_folds = 3
max_depth = 6
min_size = 3
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores Gini: %s' % scores)
print('Mean Accuracy Gini: %.3f%%' % (sum(scores)/float(len(scores))))

scores = evaluate_algorithm(dataset, decision_tree_lib, n_folds, max_depth, min_size)
print('Scores Gini Sklearn: %s' % scores)
print('Mean Accuracy Gini Sklearn: %.3f%%' % (sum(scores)/float(len(scores))))
