# Classifying handwritten digits using KNN, SVM, and MLP
from functools import wraps
from time import time
from collections import namedtuple

from sklearn import datasets, metrics, neighbors, svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Our dataset
digits = datasets.load_digits()

# For tracking performance of each classifier
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
    images_and_labels = list(zip(digits.images, digits.target))

    for index, (image, label) in enumerate(images_and_labels[:10]):
        plt.subplot(2, 5, index + 1)
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title("{}".format(label))

    plt.show()

###############################################################################
# Classify with KNN
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
@timed
def demo_knn(train_x, test_x, train_y, test_y, classifier=None):

    knn = classifier
    if knn is None:
        knn = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    knn.fit(train_x, train_y)

    expected = test_y
    predicted = knn.predict(test_x)

    return metrics.accuracy_score(expected, predicted)

def tune_knn():

    results = {}

    for k in range(1, 10):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        performance_data = run_classifier(20, demo_knn,
                                          test_size=0.10, classifier=knn)
        save_performance_data(results, str(k) + "nn", performance_data)

    print_averages(results)
    plot_performance(results)


###############################################################################
# Classify with a Support Vector Machine
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
@timed
def demo_svm(train_x, test_x, train_y, test_y, classifier=None):

    svm_classifier = classifier
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
def demo_mlp(train_x, test_x, train_y, test_y, classifier=None):

    mlp = classifier
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

def tune_mlp():

    results = {}

    for n in range(100, 1001, 100):
        print("Working on MLP with {} hidden layers".format(n))
        mlp = MLPClassifier(hidden_layer_sizes=(n, n),
                            max_iter=400,
                            alpha=1e-4,
                            solver='lbfgs',     # works well for small datasets
                            verbose=False,
                            tol=1e-4,
                            random_state=1)

        performance_data = run_classifier(10, demo_mlp,
                                          test_size=0.10,
                                          classifier=mlp)

        save_performance_data(results, str(n) + "_mlp", performance_data)

    print_averages(results)
    plot_performance(results)

def menu_selection():

    selection = 6
    try:
        selection = int(input("""
                Please make a selection:
                1) Demo K-nearest neighbor (KNN)
                2) Demo Support vector machine (SVM)
                3) Demo Multi-layer perceptron (MLP)
                4) Demo All
                5) Plot dataset sample
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

def run_classifier(iterations, func, test_size=0.10, classifier=None):

    performance_data = []

    for i in range(iterations):
        # Split data set into training and test sets
        train_x, test_x, train_y, test_y = train_test_split(digits.data,
                                                            digits.target,
                                                            test_size=test_size)

        accuracy, time_elapsed = func(train_x, test_x, train_y, test_y, classifier)
        performance_data.append(PerformanceData(accuracy, time_elapsed))

    return performance_data

def extract_performance_data(performance_data, metric):
    metrics = {"score": 0, "time_elapsed": 1}
    return [data[metrics[metric]] for data in performance_data]

def compute_averages(performanceDataPoints):

    scores = extract_performance_data(performanceDataPoints, "score")
    avg_score = sum(scores) / len(performanceDataPoints)

    times = extract_performance_data(performanceDataPoints, "time_elapsed")
    avg_time_elapsed = sum(times) / len(performanceDataPoints)

    return avg_score, avg_time_elapsed

def print_averages(results):

    for classifier, performance in results.items():
        avg_score = round(performance["avg_score"], 6)
        avg_time_elapsed = round(performance["avg_time_elapsed"], 2)
        print("{} score {}% in {}s".format(classifier,
                                           avg_score,
                                           avg_time_elapsed))

def autolabel(ax, bars):

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                round(height, 2), ha='center', va='bottom')


def plot_average_time_elapsed(results):

    classifiers = list(results.keys())
    avg_time_elapsed = [results[c]["avg_time_elapsed"] for c in classifiers]

    fig, ax = plt.subplots()
    ax.set_title("Average Time Elapsed Learning the Digits Data set")
    ax.set_ylabel("Average Time Elapsed (seconds)")

    y_position = range(len(classifiers))
    classifier_bars = ax.bar(y_position, avg_time_elapsed, align="center")

    ax.set_xticks(y_position)
    ax.set_xticklabels(classifiers)

    autolabel(ax, classifier_bars)
    plt.show(block=False)

def plot_average_score(results):

    classifiers = list(results.keys())
    avg_score = [results[c]["avg_score"] for c in classifiers]

    fig, ax = plt.subplots()
    ax.set_title("Average Classifer Score on Digits Data set")
    ax.set_ylabel("Average Score")

    y_position = range(len(classifiers))
    classifier_bars = ax.bar(y_position, avg_score, align="center")

    ax.set_xticks(y_position)
    ax.set_xticklabels(classifiers)

    autolabel(ax, classifier_bars)
    plt.show(block=False)

def gather_demos(selection):

    demos = []

    if selection == 1:
        demos.append(("KNN", demo_knn))
    elif selection == 2:
        demos.append(("SVM", demo_svm))
    elif selection == 3:
        demos.append(("MLP", demo_mlp))
    else:
        demos.append(("KNN", demo_knn))
        demos.append(("SVM", demo_svm))
        demos.append(("MLP", demo_mlp))

    return demos


def save_performance_data(results, classifier, performance_data):

    avg_score, avg_time_elapsed = compute_averages(performance_data)
    results[classifier] = {"avg_score": avg_score,
                            "avg_time_elapsed": avg_time_elapsed,
                            "performance_data": performance_data}


def gather_results(demos, iterations):

    results = {}

    for unit in demos:
        classifier, fn = unit
        performance_data = run_classifier(iterations, fn, digits, test_size=0.10)
        save_performance_data(results, classifier, performance_data)
    return results

def plot_performance(results):
    plot_average_score(results)
    plot_average_time_elapsed(results)

def run_tests(selection):

    iterations = num_iterations()

    print("Training classifiers on {} digit samples for {} iterations.\n"
          .format(len(digits.data), iterations))

    demos = gather_demos(selection)
    results = gather_results(demos, iterations)

    print_averages(results)
    plot_performance(results)

def main():

    try:

        selection = 0
        while selection != 6:

            selection = menu_selection()
            if selection <= 4:
                run_tests(selection)
            elif selection == 5:
                visualize_data()
            elif selection == 7:
                tune_knn()
            elif selection == 8:
                tune_mlp()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()


