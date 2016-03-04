import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

iris = datasets.load_iris()

n_samples, n_features = iris.data.shape

x = 2
y = 3
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.scatter(iris.data[:, x], iris.data[:, y], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x])
plt.ylabel(iris.feature_names[y])
plt.show()