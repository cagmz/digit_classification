# Classifying handwritten digits using KNN, SVM, and MLP
from sklearn import datasets, metrics, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from functools import wraps
from time import time
from collections import namedtuple

PerformanceData = namedtuple("PerformanceData", ["score", "time_elapsed"])

# A decorator used for timing functions
def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        results = f(*args, **kwds)
        elapsed = time() - start
        return results, elapsed
    return wrapper

def visualize_data():
    import matplotlib.pyplot as plt
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:9]):
        plt.subplot(2, 4, index + 1)
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title("Training: {}".format(label))

    plt.show()

###############################################################################
# Classify with KNN
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
@timed
def demo_knn(train_x, test_x, train_y, test_y, knn=None):
    if knn is None:
        knn = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    knn.fit(train_x, train_y)

    expected = test_y
    predicted = knn.predict(test_x)

    return metrics.accuracy_score(expected, predicted)

###############################################################################
# Classify with a Support Vector Machine
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
@timed
def demo_svm(train_x, test_x, train_y, test_y, svm_classifier=None):
    # Tweaking gamma provides different performance
    if svm_classifier is None:
        svm_classifier = svm.SVC(gamma=0.0001)

    # To apply a classifier on this data, the image must be flattened
    # and turned into in a (samples, feature) matrix
    train_x = train_x.reshape(len(train_x), -1)

    svm_classifier.fit(train_x, train_y)
    expected = test_y
    predicted = svm_classifier.predict(test_x)

    return metrics.accuracy_score(expected, predicted)

###############################################################################
# Classify with Multi-layer Perceptron
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
@timed
def demo_mlp(train_x, test_x, train_y, test_y, mlp=None):
    if mlp is None:
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

    return metrics.accuracy_score(expected, predicted)

def menu_selection():
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
    print()
    return selection

def num_iterations():
    iterations = 1
    try:
        iterations = int(input("""
                Please enter the number of iterations:
                > """))
    except ValueError:
        pass
    print()
    return iterations

def run_classifier(iterations, func, dataset, test_size=0.10):

    performance_data = []
    for i in range(iterations):
        # Split data set into training and test sets
        train_x, test_x, train_y, test_y = train_test_split(dataset.data,
                                                            dataset.target,
                                                            test_size=test_size)
        accuracy, time_elapsed = func(train_x, test_x, train_y, test_y)
        performance_data.append(PerformanceData(accuracy, time_elapsed))

    return performance_data

def compute_averages(performanceDataPoints, round_to=2):
    avg_score = sum(dataPoint.score for dataPoint in performanceDataPoints) / len(performanceDataPoints)
    avg_time_elapsed = sum(dataPoint.time_elapsed for dataPoint in performanceDataPoints) / len(performanceDataPoints)
    return round(avg_score, round_to), round(avg_time_elapsed, round_to)

def run_tests(selection):
    digits = datasets.load_digits()
    iterations = num_iterations()

    print("Training classifiers on {} handwritten digit samples for {} iterations."
          .format(len(digits.data), iterations))

    work_units = []
    if selection == 1:
        work_units.append(("KNN", demo_knn))
    elif selection == 2:
        work_units.append(("SVM", demo_svm))
    elif selection == 3:
        work_units.append(("MLP", demo_mlp))
    else:
        print("Called")
        work_units.append(("KNN", demo_knn))
        work_units.append(("SVM", demo_svm))
        work_units.append(("MLP", demo_mlp))

    results = {}
    for unit in work_units:
        classifier, fn = unit
        performance_data = run_classifier(iterations, fn, digits, test_size=0.10)
        avg_accuracy, avg_time_elapsed = compute_averages(performance_data)
        results[classifier] = (results.get(classifier, 0) + avg_accuracy,
                                avg_time_elapsed)

    for classifier, averages in results.items():
        avg_score, avg_time = round(averages[0], 2), round(averages[1], 2)
        print("{} score {}% in {}s".format(classifier, avg_score, avg_time))

def main():
    try:
        selection = 0
        while selection != 6:

            selection = menu_selection()
            if selection <= 4:
                run_tests(selection)
            elif selection == 5:
                visualize_data()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()


