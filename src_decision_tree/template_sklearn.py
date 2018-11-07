from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()  #
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, y_train)

# dot_data = tree.export_graphviz(clf, out_file="/tmp/img/Iris_Decision_Tree_sklean.dot")

y_hat = clf.predict(X_test, check_input=True)

print(accuracy_score(y_test, y_hat))
