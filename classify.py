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
    friendly_fn_name, results = f(*args, **kwds)
    elapsed = time() - start
    results[friendly_fn_name].append(elapsed)
    return results
  return wrapper

def visualize_data(digits):
    import matplotlib.pyplot as plt

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:8]):
        plt.subplot(2, 4, index + 1)
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title("Training: {}".format(label))

    plt.show()

###############################################################################
# Classify with KNN
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
@timed
def demo_knn(train_x, test_x, train_y, test_y, results):
    knn = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn.fit(train_x, train_y)

    expected = test_y
    predicted = knn.predict(test_x)

    results["KNN"] = [metrics.accuracy_score(expected, predicted)]
    return "KNN", results

###############################################################################
# Classify with a Support Vector Machine
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
@timed
def demo_svm(train_x, test_x, train_y, test_y, results):
    # Tweaking gamma provides different performance
    svm_classifier = svm.SVC(gamma=0.0001)

    # To apply a classifier on this data, the image must be flattened
    # and turned into in a (samples, feature) matrix
    train_x = train_x.reshape(len(train_x), -1)

    svm_classifier.fit(train_x, train_y)
    expected = test_y
    predicted = svm_classifier.predict(test_x)

    results["SVM"] = [metrics.accuracy_score(expected, predicted)]
    return "SVM", results

###############################################################################
# Classify with Multi-layer Perceptron
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
@timed
def demo_mlp(train_x, test_x, train_y, test_y, results):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100),
                        max_iter=400,
                        alpha=1e-4,
                        solver='lbfgs',     # works well for small datasets
                        verbose=False,
                        tol=1e-4,
                        random_state=1)

    mlp.fit(train_x, train_y)
    expected = test_y
    predicted = mlp.predict(test_x)

    results["MLP"] = [metrics.accuracy_score(expected, predicted)]
    return "MLP", results

def get_input():
    selection = 6
    try:
        selection = int(input("""
                Please make a selection:
                1) Demo K-nearest neighbor (KNN)
                2) Demo Support vector machine (SVM)
                3) Demo Multi-layer perceptron (MLP)
                4) Demo All
                5) Plot sample data (requires matplotlib)
                6) Quit\n
                > """))
    except ValueError:
        pass
    return selection

def main():
    try:
        digits = datasets.load_digits()
        print("Training classifiers on {} handwritten digit samples.".format(len(digits.data)))

        demos = [demo_knn, demo_svm, demo_mlp]

        selection = 0
        while selection != 6:

            # Split data set into training and test sets
            train_x, test_x, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.10)

            results = {}
            if selection == 1:
                demos[0](train_x, test_x, train_y, test_y, results)
            elif selection == 2:
                demos[1](train_x, test_x, train_y, test_y, results)
            elif selection == 3:
                demos[2](train_x, test_x, train_y, test_y, results)
            elif selection == 4:
                [d(train_x, test_x, train_y, test_y, results) for d in demos]
            elif selection == 5:
                visualize_data(digits)

            for classifier, output in results.items():
                accuracy, elapsed = round(output[0], 6), round(output[1], 6)
                print("{} score {}% in {}s".format(classifier, accuracy, elapsed))

            selection = get_input()
            print()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

