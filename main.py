import numpy as np
import sns
from matplotlib.colors import ListedColormap
import pandas as pd
from matplotlib.pyplot import xlabel
from numpy.ma.core import identity
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import copy

best_neurons_0_8 = None
best_neurons_0_2 = None
max_iter = 100000
jd =1

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
    plt.ylim(0, 1.05)
    plt.title(title)
    if xlabel != 'epochs':
        for param in param_range:
            plt.axvline(x=param, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    #plt.grid()
    plt.show()


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
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
        # label = label.tointeger()
        plt.scatter(X[y == label, 0], X[y == label, 1], color=colors[int(label)], edgecolor='k')
    # plt.legend()
    plt.title(title)
    plt.show()


def experiment_1_svm(X, y, kernel, i, j):
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # C_values = np.logspace(-2, 2, 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []

    best_acc = 0
    best_model = None
    best_C = 0

    ##################################################################33
    start_value = -2
    stop_value = 6
    num_values = stop_value - start_value + 1
    C_values = np.logspace(start_value, stop_value, num_values)
    ##################################################################33

    #C_values = np.arange(0.01, 11.0, 0.1)
    for C in C_values:
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            best_model = model
            best_C = C
    # print(train_scores)
    # plot_accuracy(train_scores, test_scores, C_values, 'C', f'SVM ({kernel}) - Accuracy vs C')

    plot_decision_boundary(X_train, y_train, best_model,
                           f'Data: 2_{j}.csv | SVM ({kernel}) | Best C={best_C.round(3)} Accuracy={best_acc:.3f} - Train')
    plot_decision_boundary(X_test, y_test, best_model,
                           f'Data: 2_{j}.csv | SVM ({kernel}) | Best C={best_C.round(3)} Accuracy={best_acc:.3f} - Test')

    # y_pred = best_model.predict(X_test)
    # plot_confusion(y_test, y_pred, 'Confusion Matrix - Best SVM')


def experiment_1_mlp(X, y, activation, j):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []

    best_acc = 0
    best_model = None
    best_size = 0

    #hidden_layer_sizes_list = [1, 3, 5, 10, 20, 50, 100]
    hidden_layer_sizes_list = np.arange(2, 64, 4)
    for size in hidden_layer_sizes_list:
        model = MLPClassifier(hidden_layer_sizes=(size,), activation=activation, max_iter=max_iter, solver='sgd', tol=0,
                              n_iter_no_change=max_iter)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            best_model = model
            best_size = size

    # plot_accuracy(train_scores, test_scores, hidden_layer_sizes_list, 'Hidden Layer Size', f'MLP ({activation}) - Accuracy vs Neurons')

    plot_decision_boundary(X_train, y_train, best_model,
                           f'Data: 2_{j}.csv | MLP ({activation}) | neurons={best_size} (best) Accuracy={best_acc:.3f} - Train')
    plot_decision_boundary(X_test, y_test, best_model,
                           f'Data: 2_{j}.csv | MLP ({activation}) | neurons={best_size} (best)best Accuracy={best_acc:.3f} - Test')

    # y_pred = best_model.predict(X_test)
    # plot_confusion(y_test, y_pred, 'Confusion Matrix - Best MLP')


def experiment_2_and_3_knn(X, y, experiment, j):
    if experiment == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []
    models = []

    best_acc = 0
    best_model = None
    best_n = 0

    first_value = 1
    last_value = 15
    n_neighbors_list = np.arange(first_value, last_value, 1)
    for n in n_neighbors_list:
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        models.append(model)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_n = n

    plot_accuracy(train_scores, test_scores, n_neighbors_list, 'n_neighbors', f'Data: 2_{j}.csv | KNN')

    # data: train
    plot_decision_boundary(X_train, y_train, models[0], f'Data: 2_{j}.csv | KNN | n_neighbors={first_value} | Train')
    y_pred = models[0].predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | KNN | n_neighbors={first_value} | Train')

    plot_decision_boundary(X_train, y_train, best_model, f'Data: 2_{j}.csv | KNN | n_neighbors={best_n} | Train')
    y_pred = best_model.predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | KNN | n_neighbors={best_n} | Train')

    plot_decision_boundary(X_train, y_train, models[-1],
                           f'Data: 2_{j}.csv | KNN | n_neighbors={last_value - 1} | Train')
    y_pred = models[-1].predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | KNN | n_neighbors={last_value - 1} | Train')

    # data: test
    plot_decision_boundary(X_test, y_test, models[0], f'Data: 2_{j}.csv | KNN | n_neighbors={first_value} | Test')
    y_pred = models[0].predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | KNN | n_neighbors={first_value} | Test')

    plot_decision_boundary(X_test, y_test, best_model, f'Data: 2_{j}.csv | KNN | n_neighbors={best_n} | Test')
    y_pred = best_model.predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | KNN | n_neighbors={best_n} | Test')

    plot_decision_boundary(X_test, y_test, models[-1],
                           f'Data: 2_{j}.csv | KNN | n_neighbors={last_value - 1} | Test')
    y_pred = models[-1].predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | KNN | n_neighbors={last_value - 1} | Test')


def experiment_2_and_3_svm(X, y, experiment, j):
    if experiment == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []
    models = []

    best_acc = 0
    best_model = None
    best_C = 0

    start_value = -2
    stop_value = 6
    num_values = stop_value - start_value + 1
    C_values = np.logspace(start_value, stop_value, num_values)
    first_value = C_values[0]
    last_value = C_values[-1]
    # C_values = np.arange(first_value, last_value, 0.1)
    for C in C_values:
        model = SVC(C=C, kernel='rbf')
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        models.append(model)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_C = C
    ##################################3333
    x_label = 'log(C)'
    info = 'SVM | C'
    range_x = np.arange(start_value, stop_value + 1)
    print(C_values)
    print(range_x)
    plot_accuracy(train_scores, test_scores, range_x, x_label, f'Data: 2_{j}.csv | SVM')

    # data: train
    plot_decision_boundary(X_train, y_train, models[0], f'Data: 2_{j}.csv | SVM | C={first_value} | Train')
    y_pred = models[0].predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={first_value} | Train')

    plot_decision_boundary(X_train, y_train, best_model, f'Data: 2_{j}.csv {info}={best_C} | Train')
    y_pred = best_model.predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={best_C} | Train')

    plot_decision_boundary(X_train, y_train, models[-1],
                           f'Data: 2_{j}.csv | {info}={last_value} | Train')
    y_pred = models[-1].predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={last_value} | Train')

    # data: test
    plot_decision_boundary(X_test, y_test, models[0], f'Data: 2_{j}.csv | {info}={first_value} | Test')
    y_pred = models[0].predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={first_value} | Test')

    plot_decision_boundary(X_test, y_test, best_model, f'Data: 2_{j}.csv | {info}={best_C} | Test')
    y_pred = best_model.predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={best_C} | Test')

    plot_decision_boundary(X_test, y_test, models[-1],
                           f'Data: 2_{j}.csv | {info}={last_value} | Test')
    y_pred = models[-1].predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={last_value} | Test')

def experiment_2_and_3_mlp(X, y, experiment, j):
    if experiment == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []
    models = []

    best_acc = 0
    best_model = None
    best_size = 0

    #hidden_layer_sizes_list = [1, 3, 5, 10, 20, 50, 100]
    hidden_layer_sizes_list = np.arange(2, 64, 4)
    for size in hidden_layer_sizes_list:
        model = MLPClassifier(hidden_layer_sizes=(size,), activation='relu', max_iter=max_iter, solver='sgd', tol=0,
                              n_iter_no_change=max_iter)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        models.append(model)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_size = size

        # plot_decision_boundary(X_train, y_train, best_model,
        #                        f'Data: 2_{j}.csv | MLP | Best size={best_size} Accuracy={best_acc:.3f} - Train')
        # plot_decision_boundary(X_test, y_test, best_model,
        #                        f'Data: 2_{j}.csv | MLP ({activation}) | Best size={best_size} Accuracy={best_acc:.3f} - Test')

    global best_neurons_0_8, best_neurons_0_2
    if experiment == 1:
        best_neurons_0_8 = best_size
    else:
        best_neurons_0_2 = best_size

    x_label = 'hidden_layer_size'
    info = 'MLP | neurons'
    plot_accuracy(train_scores, test_scores, hidden_layer_sizes_list, x_label, f'Data: 2_{j}.csv | MLP')

    # data: train
    plot_decision_boundary(X_train, y_train, models[0], f'Data: 2_{j}.csv | MLP | neurons={hidden_layer_sizes_list[0]} | Train')
    y_pred = models[0].predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[0]} | Train')

    plot_decision_boundary(X_train, y_train, best_model, f'Data: 2_{j}.csv {info}={best_size} | Train')
    y_pred = best_model.predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={best_size} | Train')

    plot_decision_boundary(X_train, y_train, models[-1],
                           f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[-1]} | Train')
    y_pred = models[-1].predict(X_train)
    plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[-1]} | Train')

    # data: test
    plot_decision_boundary(X_test, y_test, models[0], f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[0]} | Test')
    y_pred = models[0].predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[0]} | Test')

    plot_decision_boundary(X_test, y_test, best_model, f'Data: 2_{j}.csv | {info}={best_size} | Test')
    y_pred = best_model.predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | K{info}={best_size} | Test')

    plot_decision_boundary(X_test, y_test, models[-1],
                           f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[-1]} | Test')
    y_pred = models[-1].predict(X_test)
    plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={hidden_layer_sizes_list[-1]} | Test')

def experiment_4_mlp(X, y, experiment, j, random_state, iteration):
    if experiment == 1:
        #train_size = 0.8
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=random_state)
        hidden_layer_size = (best_neurons_0_8,)
        #hidden_layer_size = (6,)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=0.2, random_state=random_state)
        #train_size = 0.2
        hidden_layer_size = (best_neurons_0_2,)
        #hidden_layer_size = (6,)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_scores = []
    test_scores = []
    models = []

    print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAa{hidden_layer_size}')

    best_acc = 0
    train_acc_for_best_test_acc = 0
    best_model = None
    best_epoch = 0
    first_epoch_model = None
    last_epoch_model = None


    model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation='relu', max_iter=1, solver='sgd', tol=0,
                              n_iter_no_change=1, warm_start=True)

    first_epoch = 1
    last_epoch = max_iter
    epochs= range(first_epoch, last_epoch + 1)
    classes = np.unique(y)
    for epoch in epochs:
        model.partial_fit(X_train, y_train, classes=classes)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        # train_acc = model.score(X_train, y_train)
        # test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        if epoch == first_epoch:
            first_epoch_model = copy.deepcopy(model)

        if epoch == last_epoch:
            last_epoch_model = copy.deepcopy(model)

        if test_acc > best_acc:
            best_acc = test_acc
            train_acc_for_best_test_acc = train_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        # plot_decision_boundary(X_train, y_train, best_model,copy.deepcopy(model)copy.deepcopy(model)
        #                        f'Data: 2_{j}.csv | MLP | Best size={best_size} Accuracy={best_acc:.3f} - Train')
        # plot_decision_boundary(X_test, y_test, best_model,
        #                        f'Data: 2_{j}.csv | MLP ({activation}) | Best size={best_size} Accuracy={best_acc:.3f} - Test')


    if random_state == 42:
        x_label = 'epochs'
        info = 'MLP | epoch'
        plot_accuracy(train_scores, test_scores, epochs, x_label, f'Data: 2_{j}.csv | MLP')

        # data: train
        plot_decision_boundary(X_train, y_train, first_epoch_model, f'Data: 2_{j}.csv | MLP | epoch={epochs[0]} | Train')
        #y_pred = models[0].predict(X_train)
        #plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={epochs[0]} | Train')

        plot_decision_boundary(X_train, y_train, best_model, f'Data: 2_{j}.csv {info}={best_epoch} | Train')
        # y_pred = best_model.predict(X_train)
        # plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={best_size} | Train')

        plot_decision_boundary(X_train, y_train, last_epoch_model,
                               f'Data: 2_{j}.csv | {info}={epochs[-1]} | Train')
        # y_pred = models[-1].predict(X_train)
        # plot_confusion(y_train, y_pred, f'Data: 2_{j}.csv | {info}={epochs[-1]} | Train')

        # data: test
        plot_decision_boundary(X_test, y_test, first_epoch_model, f'Data: 2_{j}.csv | {info}={epochs[0]} | Test')
        # y_pred = models[0].predict(X_test)
        # plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={epochs[0]} | Test')

        plot_decision_boundary(X_test, y_test, best_model, f'Data: 2_{j}.csv | {info}={best_epoch} | Test')
        # y_pred = best_model.predict(X_test)
        # plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | K{info}={best_size} | Test')

        plot_decision_boundary(X_test, y_test, last_epoch_model,
                               f'Data: 2_{j}.csv | {info}={epochs[-1]} | Test')
        # y_pred = models[-1].predict(X_test)
        # plot_confusion(y_test, y_pred, f'Data: 2_{j}.csv | {info}={epochs[-1]} | Test')
    else:
        print(f'\niteration: {iteration} | experiment: {experiment} | random_state: {random_state}')
        print('\ntrain accuracies: ')
        print(f'first epoch: {train_scores[0]}')
        print(f'best epoch ({best_epoch}): {train_acc_for_best_test_acc}')
        print(f'last epoch: {train_scores[-1]}')
        print('\ntest accuracies: ')
        print(f'first epoch: {test_scores[0]}')
        print(f'best epoch ({best_epoch}): {best_acc}')
        print(f'last epoch: {test_scores[-1]}')
        print('======================================================')


if __name__ == "__main__":
    for j in range(1, 4):
        X, y_true = load(f'data/2_{j}.csv')
        experiment_1_svm(X, y_true, 'linear', 2, j)
        experiment_1_svm(X, y_true, 'rbf', 2, j)
        experiment_1_mlp(X, y_true, 'identity', j)
        experiment_1_mlp(X, y_true, 'relu', j)
    for j in range(2, 4):
        X, y_true = load(f'data/2_{j}.csv')
        experiment_2_and_3_knn(X, y_true, 1, j)
        experiment_2_and_3_svm(X, y_true, 1, j)
        experiment_2_and_3_mlp(X, y_true, 1, j)
        experiment_2_and_3_knn(X, y_true, 2, j)
        experiment_2_and_3_svm(X, y_true, 2, j)
        experiment_2_and_3_mlp(X, y_true, 2, j)

    X, y_true = load(f'data/2_3.csv')
    experiment_4_mlp(X, y_true, 1, 3, 42, 0)
    experiment_4_mlp(X, y_true, 2, 3, 42, 0)

    random_states = np.random.randint(0, 200, size=10)
    for i in range(10):
        experiment_4_mlp(X, y_true, 1, 3, random_states[i], i + 1)
    for i in range(10):
        experiment_4_mlp(X, y_true, 2, 3, random_states[i], i + 1)
