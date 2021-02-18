from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import neighbors, datasets
import numpy as np
import math
import random
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import time
from sklearn.model_selection import validation_curve
from sklearn.model_selection import validation_curve
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
    plt.yticks(np.arange(.85, .92, .02))
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()
    plt.figure()



def plotLRLC(train_x, train_y):

    parameter_range = range(1, 200, 10)
    lrs = [.0005,.001,.005]
    colors = ['green', 'blue', 'red']
    for x in range(len(lrs)):
        train_score, test_score = validation_curve(
            MLPClassifier(random_state=42, hidden_layer_sizes=(20,), alpha=4, learning_rate_init=lrs[x]), train_x, train_y,
            param_name="max_iter",
            param_range=parameter_range,
            cv=5, scoring="accuracy")

        plt.plot(parameter_range, np.mean(train_score, axis=1),
                 label='Train LR ={0}'.format(lrs[x]), color=colors[x])
        plt.plot(parameter_range, np.mean(test_score, axis=1), '--',
                 label='CV LR ={0}'.format(lrs[x]), color=colors[x])

    plt.title("Learning Rate Learning Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.ylim(.77, .90)
    plt.tight_layout()
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig('ReportNNBD LR LC.png')
    plt.close()
    plt.figure()



def main(alpha=0, layers = None, lr = 0, mi =0, train_x=None, test_x=None, train_y=None, test_y=None):

    random.seed(42)
    plt.grid(True)

    if alpha == 0:
        np.random.seed(26)
        datafile = "Data/biodeg.csv"
        data = np.genfromtxt(datafile, delimiter=";")

        data = data[:910, :]
        data = np.delete(data, slice(650, 820), 0)
        np.random.shuffle(data)

        X = data[:, :-1]
        y = data[:, -1]

        # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

        # from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
        sc = StandardScaler()
        train_x = sc.fit_transform(train_x)
        test_x = sc.transform(test_x)


    # the code to call the Validation Curve method was taken from https://www.geeksforgeeks.org/validation-curve/
    # the code to call the Learning Curve method was taken from https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

    # train_sizes, train_scores, test_scores = learning_curve(MLPClassifier( random_state=42), train_x, train_y,cv=5,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title = "NN Learning Curve Default params", name = "ReportNNBD LC 1" )
    #
    # parameter_range = [.0001, .001,  .005, .01, .1, 1, 3, 5,10]#range(1,11)
    # train_score, test_score = validation_curve(MLPClassifier(random_state=42), train_x, train_y,
    #                                            param_name="alpha",
    #                                            param_range=parameter_range,
    #                                            cv=5, scoring="accuracy")
    #
    # plotValidationCurves(train_score, test_score, title="NN alpha Validation Curve", name="ReportNNBD VC 1",
    #                      param_name="alpha", parameter_range = parameter_range)


    # # parameter_range = [(50,100,), (20, 50), (10,30), (30,70),(100,100), (10,20), (20,20), (150,)]
    # parameter_range = [(10, 10,), (10, 20), (10, 30), (10, 50), (10, 100), (10, 150)]
    # # parameter_range = [(1, ),(10,),(20,), (30,), (50,), (100,), (150)]
    # # parameter_range = ['(10,10)', '(10,20)', '(10,30)', '(10,40)', '(10,50)','(5,10)', '(5,20)', '(50,)', '(100,)']
    # train_score, test_score = validation_curve(MLPClassifier(random_state=42, alpha=4), train_x, train_y,
    #                                            param_name="hidden_layer_sizes",
    #                                            param_range=parameter_range,
    #                                            cv=5, scoring="accuracy")
    # parameter_range = ['(10, 10,)', '(10, 20)', '(10, 30)', '(10, 50)', '(10, 100)', '(10, 150)']
    # # parameter_range = ['(1,)', '(10,)', '(20,)', '(30,)', '(50,)', '(100,)', '(150)']
    # plotValidationCurves(train_score, test_score, title="NN Validation Curve Layers alpha = 4", name="ReportNNBD VC layer 2",
    #                      param_name="# nodes", parameter_range=parameter_range)


    # plotLRLC(train_x, train_y)


    # train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(random_state=42, hidden_layer_sizes = (20,), alpha = 4, learning_rate_init=.005, max_iter=30), train_x, train_y,cv=5,
    #                                                         scoring='accuracy',
    #                                                         n_jobs=-1)
    #
    # plotLearningCurves(train_sizes, train_scores, test_scores, title = "NN Learning Curve - Tuned ", name = "ReportNNBD LC 2 final" )
    #



    print("NN")
    best_est = MLPClassifier(alpha = alpha , hidden_layer_sizes=layers, random_state=42, learning_rate_init=lr, max_iter=mi)
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