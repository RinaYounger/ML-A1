import pandas as pd
import seaborn as sns; sns.set()
from sklearn.datasets import make_classification
import random
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import time
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate


# From https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
def plotLearningCurves(train_sizes, train_scores, test_scores, title, name):

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, color="g", label="Cross-validation score")

    # # Draw bands
    # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    # plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.yticks(np.arange(.7, 1, .02))
    plt.tight_layout()
    plt.savefig(name)
    plt.close()
    plt.figure()

#From https://www.geeksforgeeks.org/validation-curve/
def plotValidationCurves(train_score, test_score, title, name, param_name, parameter_range):

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score,
             label="Training Score", color='b')
    plt.plot(parameter_range, mean_test_score,
             label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()
    plt.figure()





# From https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
def plotKernelsGamma(train_x, train_y):

    parameter_range =[.001, 0.01, .1]
    Cs = [10,1,1]
    train_colors = ['green', 'blue', 'red', 'orange', 'black', 'orange']
    test_colors = ['green', 'blue', 'red', 'orange', 'black', 'orange']

    plt.figure(figsize=(9, 5))

    for x in range (len(parameter_range)):
        train_sizes, train_scores, test_scores = learning_curve(SVC(gamma = parameter_range[x], C = Cs[x]), train_x, train_y,
                                                                cv=5,
                                                                scoring='accuracy',
                                                                n_jobs=-1)


        plt.plot(train_sizes, np.mean(train_scores, axis=1) , marker = 'o', label='Train RBF Gamma ={0}'.format(parameter_range[x]), color = train_colors[x])
        plt.plot(train_sizes, np.mean(test_scores, axis=1), '--', marker = 'o', label='CV RBF Gamma={0}'.format(parameter_range[x]), color = test_colors[x])

    for x in [2]:
        train_sizes, train_scores, test_scores = learning_curve(SVC(C=5, kernel='poly', degree=(x)), train_x, train_y,
                                                   cv=5, scoring="accuracy", n_jobs=-1)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), marker='o', label='Train Poly Degree={0}'.format(x),
                 color=train_colors[x+1])
        plt.plot(train_sizes, np.mean(test_scores, axis=1), '--', marker='o', label='CV Poly Degree={0}'.format(x),
                 color=test_colors[x+1])



    plt.title("Kernel Learning Curves")
    plt.xlabel("Train Sizes")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig('ReportSVMSpam LC 1.png')
    plt.close()
    plt.figure()




def plotTimes(train_x, train_y):
    parameter_range = parameter_range = [.001, 0.01, .1]
    train_colors = ['green', 'blue', 'red', 'orange', 'black', 'orange']
    test_colors = ['green', 'blue', 'red', 'orange', 'black', 'orange']

    results = []
    labels = []
    testtimes = []


    for x in range(len(parameter_range)):

        model = SVC(gamma=parameter_range[x], C = 1, kernel = 'rbf')
        scores = cross_validate(model, train_x, y =  train_y, cv = 5)
        results.append(np.mean(scores["fit_time"]))
        testtimes.append(np.mean(scores["score_time"]))
        labels.append('RBF Gamma = {0}'.format(parameter_range[x]))


    for x in [2]:
        model = SVC(kernel='poly', degree = x, C = 1)
        scores = cross_validate(model, train_x, y=train_y, cv=5)
        results.append(np.mean(scores["fit_time"]))
        testtimes.append(np.mean(scores["score_time"]))
        labels.append('Poly Degree = {0}'.format(x))

    plt.figure(figsize=(9, 5))
    plt.bar(labels, results)
    plt.title("Kernel Training Times")
    plt.xlabel("Kernels")
    plt.ylabel("Train Time")
    plt.savefig('RepoertSVMSpam Train Times.png')
    plt.close()
    plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, testtimes)
    plt.title("Kernel Testing Times")
    plt.xlabel("Kernels")
    plt.ylabel("Test Time")
    plt.savefig('ReportSVMSpam Test Times.png')
    plt.close()
    plt.figure()


def plotCValues(train_x, train_y):


    gammas = [.001, 0.01, .1]
    train_colors = ['green', 'blue', 'red', 'orange', 'black', 'orange']
    test_colors = ['green', 'blue', 'red', 'orange', 'black', 'orange']

    plt.figure(figsize=(9, 5))
    parameter_range = [0.001, 0.01, 0.1, 1, 5, 10]

    for x in range(len(gammas)):

        train_score, test_score = validation_curve(SVC(gamma=gammas[x], kernel='rbf'), train_x, train_y,
                                               param_name="C",
                                               param_range=parameter_range,
                                               cv=5, scoring="accuracy")
        plt.plot(parameter_range, np.mean(train_score, axis=1), marker='o',
                 label='Train RBF Gamma ={0}'.format(gammas[x]), color=train_colors[x])
        plt.plot(parameter_range, np.mean(test_score, axis=1), '--', marker='o',
                 label='CV RBF Gamma={0}'.format(gammas[x]), color=test_colors[x])

    for x in [2]:
        train_score, test_score = validation_curve(SVC (kernel='poly', degree = x), train_x, train_y,
                                                   param_name="C",
                                                   param_range=parameter_range,
                                                   cv=5, scoring="accuracy")

        plt.plot(parameter_range, np.mean(train_score, axis=1), marker='o', label='Train Poly Degree={0}'.format(x),
                 color=train_colors[x + 1])
        plt.plot(parameter_range, np.mean(test_score, axis=1), '--', marker='o', label='CV Poly Degree={0}'.format(x),
                 color=test_colors[x + 1])

    plt.title("C Validation Curves")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.ylim(.70,1)
    plt.tight_layout()
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig('ReportSVMSpam VC 1.png')
    plt.close()
    plt.figure()


def findBest(train_x, train_y):
    # from https://www.datasklr.com/select-classification-methods/k-nearest-neighbors

    param_grid = {'C': [ 0.1, .5, 1, 10, 100], 'gamma': [1, 0.1, .5, .03, 0.01, 0.001]}
    estimator = SVC()

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=1,
    )

    grid.fit(train_x, train_y)

    print(grid.best_params_)
    print(grid.best_estimator_)
    print(grid.best_score_)

    return grid.best_estimator_




def main(kernel = 'rbf', gamma = 0, c = 0, train_x = None, test_x = None, train_y = None, test_y = None):
    random.seed(42)

    if c == 0:

        np.random.seed(26)

        datafile = "Data/spambase.data"
        data = np.genfromtxt(datafile, delimiter=",")

        data = data[:-975, :]
        np.random.shuffle(data)

        X = data[:, :-1]
        y = data[:, -1]

        X = X[:, :-17]

        # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

        # from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
        sc = StandardScaler()
        train_x = sc.fit_transform(train_x)
        test_x = sc.transform(test_x)

    # the code to call the Validation Curve method was taken from https://www.geeksforgeeks.org/validation-curve/
    # the code to call the Learning Curve method was taken from https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

    # plotCValues(train_x, train_y)
    # plotKernelsGamma(train_x, train_y)
    # plotTimes(train_x, train_y)


    # gridBest = findBest(train_x, train_y)
    #
    # train_sizes, train_scores, test_scores = learning_curve(gridBest,
    #                                                         train_x, train_y,
    #                                                         cv=5,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title="KNN Learning Curve Grid Search",
    #                    name="ReportSVMBD LC GS")


    print("SVM")
    best_est = SVC(gamma=gamma, C=c, kernel = kernel)
    # cross_validate(best_est, train_x, train_y, cv=5)

    start = time.time()
    best_est.fit(train_x, train_y)
    end = time.time()

    train_time = (end - start)

    print("Training Time: ", train_time)

    start = time.time()
    train_pred = best_est.predict(train_x)
    end = time.time()
    query_time = (end - start)
    print("Query Time: ", query_time)

    train_score = accuracy_score(train_y, train_pred)
    print("Training Accuracy ", train_score)
    # print(classification_report(train_y, train_pred))

    y_pred = best_est.predict(test_x)
    test_score = accuracy_score(test_y, y_pred)
    print("Testing Accuracy ", test_score)
    print(classification_report(test_y, y_pred))

    return train_score, test_score, train_time, query_time



if __name__ == "__main__":

    main()

