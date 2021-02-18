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
from sklearn.ensemble import AdaBoostClassifier
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate



#From https://www.geeksforgeeks.org/validation-curve/
def plotLRandN(train_x, train_y):

    parameter_range = range(50,501,50)
    train_colors = ['green', 'blue', 'red']
    # test_colors = ['limegreen', 'dodgerblue', 'tomato']
    test_colors = ['green', 'blue', 'red']
    lr = [.25, 1, 1.75]
    for x in range (0,3):
        train_score, test_score = validation_curve(AdaBoostClassifier( learning_rate=lr[x],random_state=42), train_x, train_y,
                                                   param_name="n_estimators",
                                                   param_range=parameter_range,
                                                   cv=5, scoring="accuracy")
        plt.plot(parameter_range, np.mean(train_score, axis=1) , marker = 'o', label='Train LR={0}'.format(lr[x]), color = train_colors[x])
        plt.plot(parameter_range, np.mean(test_score, axis=1), '--', marker = 'o', label='CV LR={0}'.format(lr[x]), color = test_colors[x])
        print("Hi")


    plt.title("Learning Rates vs Num Estimators")
    plt.xlabel("Num Estimators")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best', prop={'size': 8})
    plt.savefig('ReportBoostSpam VC 2.png')
    plt.close()
    plt.figure()


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
    plt.xticks(parameter_range)
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()
    plt.figure()


def findBest(train_x, train_y):
    # from https://www.datasklr.com/select-classification-methods/k-nearest-neighbors

    estimator = AdaBoostClassifier(DecisionTreeClassifier())
    param_grid = {
                  "base_estimator__max_depth" : range(1,10),
                  "n_estimators": range(50,400, 50),
                  "learning_rate":[.1,.4,.7,1,1.2]
             }

    grid_search_KNN = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=1,
    )

    grid_search_KNN.fit(train_x, train_y)

    print(grid_search_KNN.best_params_)
    print(grid_search_KNN.best_estimator_)
    print(grid_search_KNN.best_score_)

    return grid_search_KNN.best_estimator_




def main(max_depth = 0, lr = 0, n = 0, train_x = None, test_x = None, train_y = None, test_y = None):

    random.seed(42)

    if n == 0:

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

    # train_sizes, train_scores, test_scores = learning_curve(AdaBoostClassifier(DecisionTreeClassifier(), random_state=42), train_x, train_y,cv=5,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title = "Boosting Learning Curve Unpruned", name = "ReportBoostSpam LC 1" )

    # parameter_range = np.arange(1,20)
    # train_score, test_score = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(), random_state=42), train_x, train_y,
    #                                            param_name="base_estimator__max_depth",
    #                                            param_range=parameter_range,
    #                                            cv=5, scoring="accuracy")
    #
    # plotValidationCurves(train_score, test_score, title="Boosting DT Depth Validation Curve", name="ReportBoostSpam VC 1",
    #                      param_name="Max Depth", parameter_range=parameter_range)

    # plotLRandN(train_x, train_y)

    # train_sizes, train_scores, test_scores = learning_curve(AdaBoostClassifier(random_state=42, n_estimators=100, learning_rate=.25),
    #                                                         train_x, train_y, cv=5,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title="Boosting Learning Curve N_Estimators = 100 and Learning Rate = .25",
    #                    name="ReportBoostSpam LC 2")
    #

    # gridBest = findBest(train_x, train_y)
    #
    # train_sizes, train_scores, test_scores = learning_curve(gridBest,
    #                                                         train_x, train_y,
    #                                                         cv=10,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title="Boosting Learning Curve Grid Search",
    #                    name="ReportBoostSpam LC GS")


    print("BOOST")
    best_est = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), learning_rate=lr, n_estimators=n)
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