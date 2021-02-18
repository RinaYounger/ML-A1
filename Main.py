import NewDT as dt1
import NewKNN as knn1
import NewSVM as svm1
import NN as nn1
import Boosting as b1
import DT2 as dt2
import KNN2 as knn2
import SVM2 as svm2
import NN2 as nn2
import Boost2 as b2
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import Boosting as b
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler



if __name__ == "__main__":

    random.seed(42)
    np.random.seed(26)
    datafile = "Data/biodeg.csv"
    data = np.genfromtxt(datafile, delimiter=";")

    data = data[:910,:]
    data = np.delete(data, slice(650,820), 0)
    np.random.shuffle(data)

    X = data[:,:-1]
    y = data[:,-1]

    # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

    # from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    labels = ['DT', 'KNN', 'SVM', 'NN', 'Adaboost']
    train_scores = []
    test_scores = []
    train_times = []
    test_times = []


    train, test, train_time, query_time = dt1.main(4, .015, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time =knn1.main(3, 1, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time =svm1.main('rbf', .05, 1, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time =nn1.main(4, (20,), .005, 30,train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time =b1.main(1, 1, 50, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)

    # Bar graph plotting from https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
    # https://pythonspot.com/matplotlib-bar-chart/
    plt.figure(figsize=(9, 5))
    plt.ylim(.84, .94)
    plt.bar(labels, train_scores)
    plt.title("Training Scores")
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy Scores")
    plt.savefig('Report Training Scores BD.png')
    plt.close()
    plt.figure()

    plt.figure(figsize=(9, 5))
    plt.ylim(.84, .94)
    plt.bar(labels, train_scores)
    plt.title("Testing Scores")
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy Scores")
    plt.savefig('Report Testing Scores BD.png')
    plt.close()
    plt.figure()


    plt.figure(figsize=(9, 5))
    plt.bar(labels, train_times)
    plt.title("Training Times")
    plt.xlabel("Classifiers")
    plt.ylabel("Training Time")
    plt.savefig('Report Training Time BD.png')
    plt.close()
    plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, test_times)
    plt.title("Query Times")
    plt.xlabel("Classifiers")
    plt.ylabel("Query Time")
    plt.savefig('Report Query Time BD.png')
    plt.close()
    plt.figure()





    random.seed(42)
    np.random.seed(26)

    datafile = "Data/spambase.data"
    data = np.genfromtxt(datafile, delimiter=",")

    data = data[:-975, :]
    np.random.shuffle(data)


    X = data[:, :-1]
    X = X[:, :-17]
    y = data[:, -1]

    # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

    #from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    labels = ['DT', 'KNN', 'SVM', 'NN', 'AdaBoost']
    train_scores = []
    test_scores = []
    train_times = []
    test_times = []

    train, test, train_time, query_time = dt2.main(7, .004, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time  = knn2.main(7, 1, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time = svm2.main('rbf', .01, 1, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time  = nn2.main( 1, (50,), .001, 75, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)
    train, test, train_time, query_time  = b2.main(1, .25, 100, train_x, test_x, train_y, test_y)
    train_scores.append(train)
    test_scores.append(test)
    train_times.append(train_time)
    test_times.append(query_time)

    #Bar graph plotting from https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
    # https://pythonspot.com/matplotlib-bar-chart/
    plt.figure(figsize=(9, 5))
    plt.bar(labels, train_scores)
    plt.ylim(.88,.95)
    plt.title("Training Scores")
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy Scores")
    plt.savefig('Report Training Scores Spam.png')
    plt.close()
    plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, train_scores)
    plt.ylim(.88, .95)
    plt.title("Testing Scores")
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy Scores")
    plt.savefig('Report Testing Scores Spam.png')
    plt.close()
    plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, train_times)
    plt.title("Training Times")
    plt.xlabel("Classifiers")
    plt.ylabel("Training Time")
    plt.savefig('Report Training Time Spam.png')
    plt.close()
    plt.figure()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, test_times)
    plt.title("Query Times")
    plt.xlabel("Classifiers")
    plt.ylabel("Query Time")
    plt.savefig('Report Query Time Spam.png')
    plt.close()
    plt.figure()
