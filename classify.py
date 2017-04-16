# Classifying handwritten digits using KNN, SVM, and neural networks

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load dataset
digits = datasets.load_digits()

# Split data set into training and test sets
X_digits, y_digits = digits.data, digits.target
n_samples = len(X_digits)
print("Working with {} samples".format(n_samples))
size = int(.9 * n_samples)
X_train, y_train = X_digits[:size], y_digits[:size]
X_test, y_test = X_digits[size:], y_digits[size:]

# Show the data we're working with
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Training: {}".format(label))
# plt.show()

###############################################################################
# Classify with KNN
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN score: {}".format(knn.score(X_test, y_test)))

###############################################################################
# Classify with a Support Vector Machine
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# To apply a classifier on this data, we need to flatten the image, to turn the data in a (samples, feature) matrix:
training_data = digits.images.reshape((n_samples, -1))

# Tweaking gamma provides different performance; gamma=0.001
classifier = svm.SVC(gamma=1/n_samples)

# train and predict
classifier.fit(training_data[:size], digits.target[:size])
expected = digits.target[size:]
predicted = classifier.predict(training_data[size:])

# Shows accuracy for each category
# print("SVM score: {}".format((metrics.classification_report(expected, predicted))))
print("SVM score: {}".format((metrics.accuracy_score(expected, predicted))))

###############################################################################
# Classify with Multi-layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1)

# Split data set into training and test sets
train_x, test_x, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.10)
mlp.fit(train_x, train_y)

print("MLP Training set score: {}".format(mlp.score(train_x, y_train)))  
print("MLP Test set score: {}".format(mlp.score(test_x, test_y)))

