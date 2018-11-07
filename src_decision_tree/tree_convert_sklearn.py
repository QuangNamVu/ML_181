import numpy as np


# CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
# TODO get data set in category basis in decision tree sklearn
# iris select pivot
# base decision tree:
class BaseDecisionTree():
    """
    Binary tree dataset in category
    """

    def __init__(self, criterion, class_weight=None):
        self.maxdepth = 0
        self.criterion = criterion
        self.class_weight = class_weight

    def fit(self, X, y):
        """fitting and building tree"""
        print(X.shape, y.shape)


n_nodes = 13

children_right = np.array([2, -1, 8, 5, -1, 7, -1, -1, 12, 11, -1, -1, -1])
children_left = np.array([1, -1, 3, 4, -1, 6, -1, -1, 9, 10, -1, -1, -1])
feature = np.array([3, -2, 3, 2, -2, 2, -2, -2, 2, 1, -2, -2, -2])

threshold = np.array([0.80000001, -2., 1.65000004, 4.95000005, -2.,
                        5.04999995, -2., -2., 5.04999995, 2.75,
                        -2., -2, -2.])

# Build the tree from children right, left
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth

while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
