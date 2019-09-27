# importing libraries

import pandas as pd
import numpy as np
import math
import operator


def get_train_test_split(dataset, split_ratio=0.8):
    datalen = len(dataset)
    shuffled = dataset.iloc[np.random.permutation(len(dataset))]
    train_data = shuffled[:int(split_ratio * datalen)]
    test_data = shuffled[int(split_ratio * datalen):]
    return train_data, test_data


def euclidean_dist(p1, p2):
    dist = 0.0
    for i in range(len(p1)-1):
        dist += pow((float(p1[i])-float(p2[i])),2)
        dist = math.sqrt(dist)
    return dist


def knn(training_dataset, test_value, k):
    distances = {}
    sort = {}

    # length = test_value.shape[1]

    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(training_dataset)):
        dist = euclidean_dist(test_value, training_dataset.iloc[x])

        distances[x] = dist
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    class_votes = {}

    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = training_dataset.iloc[neighbors[x]][-1]

        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0], neighbors


def cal_accuracy(data, test_data, K=5):
    correct = 0
    total = len(test_data)

    for i in range(total):
        pred_class, neigh = knn(data, test_data.iloc[i], K)
        print("predicted class: {:16} | actual class: {:16}".format(pred_class, test_data.iloc[i]['Name']))
        if pred_class == test_data.iloc[i]['Name']:
            correct += 1
        else:
            print("\t\t\tincorrect prediction", neigh)

    return (correct/total) * 100


if __name__ == '__main__':

    dataset = pd.read_csv('iris.csv')
    print('first 5 rows of the dataset')
    print(dataset.head(5))
    train_data, test_data = get_train_test_split(dataset, 0.9)
    print('first 5 rows of the train set')
    print(train_data.head(5))
    print('first 5 rows of the test set')
    print(test_data.head(5))
    print(cal_accuracy(train_data, test_data, 5))
