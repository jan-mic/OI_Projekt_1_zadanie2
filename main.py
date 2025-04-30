import numpy as np
import sns
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


def load(path):
    data = np.genfromtxt(path, delimiter=';')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def plot_accuracy(train_scores, test_scores, param_range, xlabel, title):
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores, label='Train Accuracy')
    plt.plot(param_range, test_scores, label='Test Accuracy')
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    unique_labels = np.unique(y)
    colors = ['orange', 'green', 'red']
    colors = colors[:len(unique_labels)]
    cmap = ListedColormap(colors)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    for label in np.unique(y):
        #label = label.tointeger()
        plt.scatter(X[y == label, 0], X[y == label, 1], color=colors[int(label)], edgecolor='k')
    # plt.legend()
    plt.title(title)
    plt.show()


def svm_experiment(X, y, kernel, i, j):
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    C_values = np.arange(0.01, 11.0, 0.1)
    #C_values = np.logspace(-2, 2, 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    train_scores = []
    test_scores = []

    best_acc = 0
    best_model = None
    best_C = 0

    for C in C_values:
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        if train_acc > best_acc:
            print(best_acc, train_acc)
            best_acc = train_acc
            best_model = model
            best_C = C
    #print(train_scores)
    #plot_accuracy(train_scores, test_scores, C_values, 'C', f'SVM ({kernel}) - Accuracy vs C')

    plot_decision_boundary(X_train, y_train, best_model, f'Data: 2_{j}.csv | SVM ({kernel}) | Best C={best_C.round(3)} Accuracy={best_acc:.3f} - Train')
    plot_decision_boundary(X_test, y_test, best_model, f'Data: 2_{j}.csv | SVM ({kernel}) | Best C={best_C.round(3)} Accuracy={best_acc:.3f} - Test')

    # y_pred = best_model.predict(X_test)
    # plot_confusion(y_test, y_pred, 'Confusion Matrix - Best SVM')

def mlp_experiment(X, y, activation, i, j):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []

    best_acc = 0
    best_model = None
    best_size = 0

    hidden_layer_sizes_list = [1, 3, 5, 10, 20, 50, 100]
    for size in hidden_layer_sizes_list:
        model = MLPClassifier(hidden_layer_sizes=(size,), activation=activation, max_iter=1000, solver='sgd', tol=0,
                              n_iter_no_change=1000)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            best_model = model
            best_size = size

    #plot_accuracy(train_scores, test_scores, hidden_layer_sizes_list, 'Hidden Layer Size', f'MLP ({activation}) - Accuracy vs Neurons')

    plot_decision_boundary(X_train, y_train, best_model, f'Data: 2_{j}.csv | MLP ({activation}) | Best size={best_size} Accuracy={best_acc:.3f} - Train')
    plot_decision_boundary(X_test, y_test, best_model, f'Data: 2_{j}.csv | MLP ({activation}) | Best size={best_size} Accuracy={best_acc:.3f} - Test')

    # y_pred = best_model.predict(X_test)
    # plot_confusion(y_test, y_pred, 'Confusion Matrix - Best MLP')


if __name__ == "__main__":
    # Eksperymenty sztuczne
    for i in range(2, 3):
        for j in range(1, 4):
            X, y_true = load(f'data/2_{j}.csv')
            # svm_experiment(X, y_true, 'linear', 2, j)
            # svm_experiment(X, y_true, 'rbf', 2, j)
            mlp_experiment(X, y_true, 'identity', i, j)
            mlp_experiment(X, y_true, 'relu', i, j)
