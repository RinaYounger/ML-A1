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
from sklearn.metrics import classification_report
import time
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


def findBest(train_x, train_y, test_y, test_x):
    param_dict = {"criterion": ['gini', 'entropy'],
                  "max_depth": range(1, 20),
                  "ccp_alpha" : [.001, .002, .003, .004, .005, .006, .007, .008, .009]}

    # HOW TO FIND BEST PARAMS
    # from https://medium.com/ai-in-plain-english/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda
    clf = DecisionTreeClassifier()
    grid = GridSearchCV(clf, param_grid=param_dict,
                        cv=5, verbose=1, n_jobs=-1)

    grid.fit(train_x, train_y)
    print(grid.best_params_)
    print(grid.best_estimator_)
    print(grid.best_score_)

    return grid.best_estimator_



def main(max_depth = 0, alpha = 0, train_x = None, test_x = None, train_y = None, test_y = None):

    random.seed(42)

    if max_depth == 0:

        np.random.seed(26)
        datafile = "Data/biodeg.csv"
        data = np.genfromtxt(datafile, delimiter=";")

        data = data[:910, :]
        data = np.delete(data, slice(650, 820), 0)
        np.random.shuffle(data)

        X = data[:, :-1]
        y = data[:, -1]

        #from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

        #from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
        sc = StandardScaler()
        train_x = sc.fit_transform(train_x)
        test_x = sc.transform(test_x)

        max_depth = 4
        alpha = .015

    #the code to call the Validation Curve method was taken from https://www.geeksforgeeks.org/validation-curve/
    #the code to call the Learning Curve method was taken from https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

    train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(), train_x, train_y,cv=10,
                                                            scoring='accuracy',
                                                            n_jobs=-1)

    plotLearningCurves(train_sizes, train_scores, test_scores, title = "DT Learning Curve Default Parameters", name = "ReportDTBD LC 1" )

    parameter_range = np.arange(1, 15, 1)
    train_score, test_score = validation_curve(DecisionTreeClassifier(), train_x, train_y,
                                               param_name="max_depth",
                                               param_range=parameter_range,
                                               cv=12, scoring="accuracy")

    plotValidationCurves(train_score, test_score, title = "DT Tree Depth Validation Curve Max Depth", name = "ReportDTBD VC 1", param_name= "Max_Depth", parameter_range= parameter_range)

    train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(max_depth = 4), train_x, train_y, cv=10,
                                                            scoring='accuracy',
                                                            n_jobs=-1)

    plotLearningCurves(train_sizes, train_scores, test_scores, title="DT Learning Curve max_depth = 4", name="ReportDTBD LC 2")

    # parameter_range = [.001, .002, .003, .004, .005, .006, .007, .008, .009, .01, .02]
    parameter_range = [.001, .005, .01, .02, .03, .04, .05]
    train_score, test_score = validation_curve(DecisionTreeClassifier(max_depth=4), train_x, train_y,
                                               param_name="ccp_alpha",
                                               param_range=parameter_range,
                                               cv=10, scoring="accuracy")

    plotValidationCurves(train_score, test_score, title="DT ccp Alpha Validation Curve md=4", name="ReportDTBD VC 2",
                         param_name="alpha", parameter_range=parameter_range)

    train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(max_depth = 4, ccp_alpha = .015), train_x, train_y, cv=10,
                                                            scoring='accuracy',
                                                            n_jobs=-1)

    plotLearningCurves(train_sizes, train_scores, test_scores, title="DT Learning Curve max_depth = 4 alpha = .015", name="ReportDTBD LC 3")



    # gridBest = findBest(train_x, train_y, test_y, test_x)
    #
    # train_sizes, train_scores, test_scores = learning_curve(gridBest,
    #                                                         train_x, train_y,
    #                                                         cv=10,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title="DT Learning Curve Grid Search",
    #                    name="ReportDTBD LC GS")

    print("DT")

    best_est = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=alpha)
    # cross_validate(best_est, train_x,train_y, cv = 10)

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