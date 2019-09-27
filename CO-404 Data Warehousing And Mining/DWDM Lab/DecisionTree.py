# importing libraries

import pandas as pd
import numpy as np
from pprint import pprint


# calculates the entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    result = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return result


# calculate the information gain = Gain(A) / SplitInfo_A(D)
def information_gain(data, split_attribute_name, target_name="class"):
    """
    calculates the information gain of the dataset for attribute A
    :param data: the dataset for which we're calculating the information gain
    :param split_attribute_name: name of the splitting attribute.
    :param target_name: name of the target / class.
    :return: the information gain
    """
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    information_gain_ = total_entropy - weighted_entropy
    return information_gain_


# the ID3 Decision tree algorithm
def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    """
    id3 algorithm
    :param data: the dataset for which we're running the id3 algorithm
    :param original_data: the original dataset needed to calculate the mode target feature value of the original dataset
    in the case the dataset delivered by the first parameter is empty
    :param features: the feature space of the dataset . This is needed for the recursive call since during the tree
    growing process we have to remove features from our dataset -> Splitting at each node
    :param target_attribute_name: the name of the target attribute
    :param parent_node_class: This is the value or class of the mode target feature value of the parent node for a
    specific node. This is also needed for the recursive call since if the splitting leads to a situation that there are
     no more features left in the feature space, we want to return the mode target feature value of the direct parent node.

    :return: The learnt decision tree
    """

    # Define the stopping criteria -> If one of this is satisfied, we want to return a leaf node

    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node -> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.

    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!

    else:
        # Set the default value for this node -> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [information_gain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best information gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest
            # information gain and then create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the
            # new parameters -> Here the recursion comes in!
            subtree = id3(sub_data, dataset, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return tree


def predict(query, tree, default=1):
    """
    prediction function for new unseen data
    :param query: a dictionary entry, {"feature_name" : value, ... , "feature_name" : value}
    :param tree: the ID3 tree
    :param default: base value
    :return: predicted class
    """

    # 1.Check for every feature in the query instance if this feature is existing in the tree.keys()
    #   for the first call, tree.keys() only contains the value for the root node -> if this value is not existing,
    #   we can not make a prediction and have to return the default value which is the majority value of the target
    #   feature
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # 2. First of all we have to take care of a important fact: Since we train our model with a database A and
            #    then show our model a unseen query it may happen that the feature values of these query are not
            #    existing in our tree model because non of the training instances has had such a value for this specific
            #    feature.
            #    For instance imagine the situation where your model has only seen animals with one to four
            #    legs - The "legs" node in your model will only have four outgoing branches (from one to four).
            #    If you now show your model a new instance (animal) which has for the legs feature the vale 5, you have
            #    to tell your model what to do in such a situation because otherwise there is no classification possible
            #    because in the classification step you try to run down the outgoing branch with the value 5 but there
            #    is no such a branch. Hence: Error and no Classification! We can address this issue with a
            #    classification value of for instance (999) which tells us that there is no classification possible or
            #    we assign the most frequent target feature value of our dataset used to train the model. Or, in for
            #    instance medical application we can return the most worse case - just to make sure...
            #    We can also return the most frequent value of the direct parent node. To make a long story short, we
            #    have to tell the model what to do in this situation.
            #    In our example, since we are dealing with animal species where a false classification is not that
            #    critical, we will assign the value 1 which is the value for the mammal species (for convenience).
            try:
                result = tree[key][query[key]]
            except:
                return default

            # 3. Address the key in the tree which fits the value for key --> Note that key == the features in the
            #    query. Because we want the tree to predict the value which is hidden under the key value (imagine you
            #    have a drawn tree model on the table in front of you and you have a query instance for which you want
            #    to predict the target feature - What are you doing? - Correct:
            #    You start at the root node and wander down the tree comparing your query to the node values. Hence you
            #    want to have the value which is hidden under the current node. If this is a leaf, perfect, otherwise
            #    you wander the tree deeper until you get to a leaf node.

            #    Though, you want to have this "something" [either leaf or sub_tree] which is hidden under the current
            #    node and hence we must address the node in the tree which == the key value from our query instance.
            #    This is done with tree[keys]. Next you want to run down the branch of this node which is equal to the
            #    value given "behind" the key value of your query instance e.g. if you find "legs" == to tree.keys()
            #    that is, for the first run == the root node. You want to run deeper and therefore you have to address
            #    the branch at your node whose value is == to the value behind key. This is done with query[key]
            #    e.g. query[key] == query['legs'] == 0 --> Therewith we run down the branch of the node with the
            #    value 0. Summarized, in this step we want to address the node which is hidden behind a specific branch
            #    of the root node (in the first run) this is done with: result = [key][query[key]]
            result = tree[key][query[key]]
            # 4. As said in the 2. step, we run down the tree along nodes and branches until we get to a leaf node.
            #    That is, if result = tree[key][query[key]] returns another tree object (we have represented this by a
            #    dict object -> that is if result is a dict object) we know that we have not arrived at a root node and
            #    have to run deeper the tree.
            #    Okay... Look at your drawn tree in front of you... what are you doing?...well, you run down the next
            #    branch... exactly as we have done it above with the slight difference that we already have passed a
            #    node and therewith have to run only a fraction of the tree --> You clever guy! That
            #    "fraction of the tree" is exactly what we have stored under 'result'.
            #    So we simply call our predict method using the same query instance (we do not have to drop any features
            #    from the query instance since for instance the feature for the root node will not be available in any
            #    of the deeper sub_trees and hence we will simply not find that feature) as well as the
            #    "reduced / sub_tree" stored in result.
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


# splitting the data into test-train splits for checking performance on unseen data
def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


# testing the tree model, get prediction accuracy
def test(data, tree):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["class"]) / len(data)) * 100, '%')


"""
Train the tree, Print the tree and predict the accuracy
"""

if __name__ == '__main__':
    # loading the dataset
    dataset = pd.read_csv('zoo.csv', names=['animal_name', 'hair', 'feathers', 'eggs', 'milk',
                                            'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
                                            'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize',
                                            'class'])

    print(dataset.head(10))

    # dropping the class column
    dataset = dataset.drop('animal_name', axis=1)
    print(dataset.head(10))

    # splitting data
    training_data = train_test_split(dataset)[0]
    testing_data = train_test_split(dataset)[1]

    # training the tree
    tree = id3(training_data, training_data, training_data.columns[:-1])

    # printing the learnt tree in the form of a dictionary
    pprint(tree)

    # get test performance of the tree
    test(testing_data, tree)

    