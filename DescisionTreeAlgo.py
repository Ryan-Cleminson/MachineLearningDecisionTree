# Assessment 2: Algorithm Implementation and Report

# Implement a Machine Learning Model and Test the Training Algorithm on Data

# Name: Ryan Cleminson, Student Number: 13555089

## Import Model Libraries

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import tree
from sklearn.datasets import load_iris # Using the iris dataset from sklearn
from sklearn.datasets import load_wine # Using the wine dataset from sklearn
from sklearn.model_selection import train_test_split # Using the iris dataset from sklearn
from sklearn.tree import DecisionTreeClassifier # For usecase of the sklearn decision tree classification
                                                # This is used for a coomparison with the developed model


# Untility Functions
# (Hide Me)

## Train/Test Split

"""
Purpose: 
- To identfy dataframe labels
- Extract X and y training and validation values respectively from a dataset
- Build a training and testing dataframe
"""
def construct_train_test_split(dataset, test_size=None , random_state=None):
    
  labels = dataset['feature_names']
  labels = labels + ['target']
  
  X_train, X_valid, y_train, y_valid = \
  train_test_split(dataset['data'], dataset['target'],test_size=test_size, random_state=random_state)

  train_df = pd.DataFrame(data= np.c_[X_train, y_train],
                  columns= labels)
  valid_df = pd.DataFrame(data= np.c_[X_valid, y_valid],
                  columns= labels)

  return train_df, valid_df

"""
Purpose: 
- Extract training and validation values from respective dataframes
"""
def extract_df_values(train_df, valid_df):
  X = train_df.drop( "target", axis = 1)
  valid_X = valid_df.drop( "target", axis = 1)
  y = train_df["target"]
  valid_y = valid_df["target"]

  return X, valid_X, y, valid_y

# The ID3 Model

class TreeNode: # A class used for each node and sub-node within a tree
  
  def __init__(self, max_node_depth=5, current_node_depth=0): # Initialises all node elements involved 
                                                              # for varying use in nodes
    """
    :parameter max_node_depth: Has a default value of 5 which limits the depth in which the
    tree nodes will build until
    :parameter current_node_depth: Is a running value of the current node depth
    """
    self.max_node_depth=max_node_depth
    self.current_node_depth = current_node_depth

    self.entropy = 0

    self.children = {} # One or many sub nodes which contain TreeNote elements
    self.decision = None
    self.feat_name_split = None # Splitting feature
    self.feat_value_split = None # Split Value - a list of possible values for categorical attributes

  def compute_entropy(self,y):
    """
    :parameter y: The data samples of a given discrete distribution
    """
    if len(y) < 2:
      return 0
    freq = np.array( y.value_counts(normalize=True) )
    return -(freq * np.log2(freq + 1e-16)).sum() # the small eps for 
    # safe numerical computation 

  def predict(self, X):
    """
    :parameter X: A subset of data from another subset
    """
    if self.decision is not None:
      return self.decision # Arbitrary decision
    else: 
      attribute_value = X[self.feat_name_split]

      # Check if there is a node associated to the current value
      if (attribute_value in self.children):
        child = self.children[attribute_value]
      else:
        # If the current value is not found then choose the child with the least entropy
        entropy_min = 10000000
        for child_index in self.children:
          if self.children[child_index].entropy < entropy_min:
            entropy_min = self.children[child_index].entropy
            child = self.children[child_index]
    return child.predict(X)

  def compute_info_gain(self, df_X, attribute, df_y):
    """
    :parameter df_X: A subset of data from another subset
    :parameter df_y: The targeted data samples of a given discrete distribution
    :parameter attribute: object of given dataset
    """
    attribute_vals = df_X[attribute].value_counts(normalize=True)
    cumulative_entropy = 0


    # Iterates through attribute sample values to compute information gain
    for attribute_val, attribute_vals_norm in attribute_vals.iteritems():
      targets = df_X[attribute].eq(attribute_val)
      target_derived_entropy = self.compute_entropy(df_y[targets])
      cumulative_entropy = cumulative_entropy + attribute_vals_norm * target_derived_entropy

    entropy = self.compute_entropy(df_y)
    return entropy - cumulative_entropy

  def fit(self, X, y):
    """
    :parameter X: V x A matrix of data
    :parameter y: V numeric targets
    """
    if len(X) == 0:

      # If the data is empty when this node is called, decide 0
      self.decision = 0

      return
    else: 

      unique_values = y.unique()

      if self.current_node_depth >= self.max_node_depth:
        # Exit Early if the depth reaches the maximum depth

        self.entropy = self.compute_entropy(y)
        self.decision = y.mean() # If the depth reaches maximum then decide the average

        return
      elif len(unique_values) == 1:

        # If there is only ome unique value then it is a leaf node
        self.decision = unique_values[0]

        return
      else:

        self.entropy = self.compute_entropy(y)
        information_gain_max = -1

        for attribute in X.keys(): # Check features to split the attribute
          # Compute the information gain of each attribute

          attribute_info_gain = self.compute_info_gain(X, attribute, y)
          # Check if the new information gain is higher then the current maximum
          # If yes then set new highest max + sets the split feature to that attribute
          if attribute_info_gain > information_gain_max:
            self.feat_name_split = attribute
            information_gain_max = attribute_info_gain

          # Deliver's the unique feature values associated with the maximum information gain attribute
          self.feat_value_split = X[self.feat_name_split].unique()

          # Creates the next tree node
          for value in self.feat_value_split:
            information = X[self.feat_name_split].eq(value) 
            self.children[value] = TreeNode(self.max_node_depth, self.current_node_depth + 1)
            self.children[value].fit(X[information], y[information])
 
class TreeID3:
  def __init__(self,max_node_depth = 5):
    self.root = TreeNode(max_node_depth)

  def fit(self, samples, target):
    self.root.fit(samples, target)

  def predict(self, sample):
    return self.root.predict(sample)

def compute_accuracy(tree, df_X, target):
  accuracy = 0
  accuracy_timeline = []
  # Iterates through each element of rows in X to find the accuracy and ...
  # record the raw accuracy over iterations
  for iteration, sample_set in df_X.iterrows():
    sample_prediction = tree.predict(sample_set)

    if target[iteration] == sample_prediction:
      accuracy = accuracy + 1
    elif target[iteration] == sample_prediction + 0.1:
      accuracy = accuracy + 0.5
    elif target[iteration] == sample_prediction - 0.1:
      accuracy = accuracy + 0.5
    else:
      pass
    accuracy_timeline.append(accuracy)
  accuracy = accuracy / len(target)
  return accuracy, accuracy_timeline

def decisionTreeClassifier_compute_accuracy(tree, X,y):
  SK_Acc = (skClassifier.predict(X) == y).sum()/len(y)
  return SK_Acc

def predict_df(tree, X):
  pred_df = [tree.predict(row) for i, row in X.iterrows()]
  return pred_df

# IRIS - Main Program

iris = load_iris()
iris_train_df, iris_valid_df = construct_train_test_split(iris, test_size=0.3, random_state=3)

iris_X, iris_valid_X, iris_y, iris_valid_y = extract_df_values(iris_train_df, iris_valid_df)

iris_Tree = TreeID3(max_node_depth=10)

iris_Tree.fit(iris_X, iris_y)

## ID3 Decision Tree Outcome - IRIS

### ID3 - Iris Dataset Accuracy

valid_accuracy, valid_accuracy_timeline = compute_accuracy(iris_Tree, iris_valid_X ,iris_valid_df['target'])
print(f"Accuracy of Validation Set: {valid_accuracy}")

train_accuracy, train_accuracy_timeline = compute_accuracy(iris_Tree, iris_X, iris_train_df['target'])
print(f"Accuracy of Training Set: {train_accuracy}")

### ID3 - Visualisation of Raw Iris Accuracy

plt.style.use('default')
plt.title('Raw iris accuracy over prediction iterations')
plt.xlabel("Time (iterations)")
plt.ylabel("Raw Accuracy")
plt.plot(valid_accuracy_timeline, label = 'raw valid accuracy');
plt.plot(train_accuracy_timeline, label = 'raw training accuracy');
plt.legend()
plt.show

## SKlearn Decision Tree - IRIS

### SKlearn - Iris Dataset Accuracy

skClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=1)
skClassifier.fit(iris_X, iris_y)

skvalid_accuracy = decisionTreeClassifier_compute_accuracy(skClassifier, iris_valid_X, iris_valid_df["target"])
print(f"Accuracy of Validation Set:{skvalid_accuracy}")

sktrain_accuracy = decisionTreeClassifier_compute_accuracy(skClassifier, iris_X, iris_train_df["target"])
print(f"Accuracy of Training Set: {sktrain_accuracy}")

### SKlearn - Iris Decision Tree Visualisation (Text-based)

text_representation = tree.export_text(skClassifier)
print(text_representation)

### SKlearn - Iris Decision Tree Visualisation (Graphical)

import matplotlib as mpl
fig = plt.figure(figsize=(25,10))
# The color indicates to which class the majority of the samples at each node belong to.
_ = tree.plot_tree(skClassifier, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True, fontsize=12)

# WINE - Main Program

wine = load_wine()
wine_train_df, wine_valid_df = construct_train_test_split(wine, test_size=0.3, random_state=3)

wine_X, wine_valid_X, wine_y, wine_valid_y = extract_df_values(wine_train_df, wine_valid_df)

wine_Tree = TreeID3()

wine_Tree.fit(wine_X,wine_y)

## ID3 Decision Tree Outcome - WINE

### ID3 - Wine Dataset Accuracy

valid_accuracy, valid_accuracy_timeline = compute_accuracy(wine_Tree, wine_valid_X ,wine_valid_df["target"])
print(f"Accuracy of Validation Set: {valid_accuracy}")

train_accuracy, train_accuracy_timeline = compute_accuracy(wine_Tree, wine_X, wine_train_df["target"])
print(f"Accuracy of Training Set: {train_accuracy}")

### ID3 - Visualisation of Raw Wine Accuracy

plt.style.use('default')
plt.title('Raw Wine Accuracy over Prediction Iterations')
plt.xlabel("Time (iterations)")
plt.ylabel("Raw Accuracy")
plt.plot(valid_accuracy_timeline, label = 'raw valid accuracy');
plt.plot(train_accuracy_timeline, label = 'raw training accuracy');
plt.legend()
plt.show

## SKlearn Decision Tree - WINE

### SKlearn - Wine Dataset Accuracy

skClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=10,random_state=2)
skClassifier.fit(wine_X, wine_y)

skvalid_accuracy = decisionTreeClassifier_compute_accuracy(skClassifier, wine_valid_X, wine_valid_df["target"])
print(f"Accuracy of Validation Set: {skvalid_accuracy}")

sktrain_accuracy = decisionTreeClassifier_compute_accuracy(skClassifier, wine_X, wine_train_df["target"])
print(f"Accuracy of Training Set: {sktrain_accuracy}")

### SKlearn - Wine Decision Tree Visualisation (Text-based)

text_representation = tree.export_text(skClassifier)
print(text_representation)

### SKlearn - Wine Decision Tree Visualisation (Graphical)

import matplotlib as mpl
fig = plt.figure(figsize=(25,10))
# The color indicates to which class the majority of the samples at each node belong to.
_ = tree.plot_tree(skClassifier, 
                   feature_names=wine.feature_names,  
                   class_names=wine.target_names,
                   label = 'all', filled=True, fontsize=12)