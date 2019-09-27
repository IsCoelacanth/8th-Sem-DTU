from random import seed
from random import randrange


def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 3)
for x in folds:
    print(x)


def confusionmatrix(actual, predicted, normalize=False):
    unique = sorted(set(actual))
    matrix = [[0 for _ in unique] for _ in unique]
    imap = {key: i for i, key in enumerate(unique)}

    for p, a in zip(predicted, actual):
        matrix[imap[p]][imap[a]] += 1

    if normalize:
        sigma = sum([sum(matrix[imap[i]]) for i in unique])
        matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
    return matrix


cm = confusionmatrix(
    [1, 1, 2, 0, 1, 1, 2, 0, 0, 1],  # actual
    [0, 1, 1, 0, 2, 1, 2, 2, 0, 2]  # predicted
)


print('actual : ', [1, 1, 2, 0, 1, 1, 2, 0, 0, 1])
print('predicted: ', [0, 1, 1, 0, 2, 1, 2, 2, 0, 2])
for x in cm:
    print(x)
