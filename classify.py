# Classifying handwritten digits using KNN, SVM, and MLP
from sklearn import datasets, metrics, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from functools import wraps
from time import time

# A decorator used for timing functions
def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time()
    result = f(*args, **kwds)
    elapsed = time() - start
    print("{} took {} to finish".format(f.__name__, elapsed))
    return result
  return wrapper

def visualize_data(digits):
    import matplotlib.pyplot as plt

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title("Training: {}".format(label))

    plt.show()

###############################################################################
# Classify with KNN
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
@timed
def demo_knn(digits):
    train_x, test_x, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.10)
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_x, train_y)

    expected = test_y
    predicted = knn.predict(test_x)

    print("KNN score: {}".format((metrics.accuracy_score(expected, predicted))))

###############################################################################
# Classify with a Support Vector Machine
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def demo_svm(digits):
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
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
def demo_mlp(digits):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100),
                        max_iter=400, alpha=1e-4,
                        solver='sgd',
                        verbose=10,
                        tol=1e-4,
                        random_state=1)

    # Split data set into training and test sets
    train_x, test_x, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.10)
    mlp.fit(train_x, train_y)
    print("MLP Training set score: {}".format(mlp.score(train_x, y_train)))
    print("MLP Test set score: {}".format(mlp.score(test_x, test_y)))

def get_input(digits):
    selection = 6
    try:
        selection = int(input("""
                Training classifiers on {} handwritten digit samples.

                Please make a selection:
                1) Demo K-nearest neighbor (KNN)
                2) Demo Support vector machine (SVM)
                3) Demo Multi-layer perceptron (MLP)
                4) Demo All
                5) Plot sample data (requires matplotlib)
                6) Quit\n
                > """.format(len(digits.data))))
    except ValueError:
        pass
    return selection

def main():
    try:
        digits = datasets.load_digits()
        selection = 0
        while selection != 6:
            if selection == 1:
                demo_knn(digits)
            elif selection == 2:
                demo_svm(digits)
            elif selection == 3:
                demo_mlp(digits)
            elif selection == 4:
                demo_knn(digits)
                demo_svm(digits)
                demo_mlp(digits)
            elif selection == 5:
                visualize_data(digits)

            selection = get_input(digits)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()


