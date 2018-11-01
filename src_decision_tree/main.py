from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris() #
X, y = iris.data, iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_hat = predict(X, check_input=True)
