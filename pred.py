"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy as np
import pandas

h1_weights= np.load("h1_weights.npy")
h1_bias= np.load("h1_bias.npy")
h2_weights= np.load("h2_weights.npy")
h2_bias= np.load("h2_bias.npy")

vocab = {}
with open("vocab.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        word, index =row
        vocab[word] = int(index)

# print(vocab)
def clean_text(text):

    return text.lower().split()

def make_input_vector(t, vocab):

    x =np.zeros([len(vocab),])
    for word in clean_text(t):

        if word in vocab:
            x[vocab[word]] += 1
    return x

def relu(x):

    return np.maximum(0, x)

def softmax(z):

    m = np.max(z)
    e =np.exp(z - m)

    return e / np.sum(e)

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the three choices: 'Pizza', 'Shawarma', 'Sushi'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the food items!
    # y = random.choice(['Pizza', 'Shawarma', 'Sushi'])

    x = make_input_vector(x, vocab)

    h1 = relu(np.dot(h1_weights, x)+ h1_bias)
    output = np.dot(h2_weights, h1) + h2_bias
    probabilities = softmax(output)
    predict_class = np.argmax(probabilities)
    final_labels = ['Pizza', 'Shawarma', 'Sushi']

    return final_labels[predict_class]

    # return final_labels[predict_class], probabilities


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    data = csv.DictReader(open(filename))
    predictions = []

    for dp in data:
        
        row_values = []

        for c in data.fieldnames:
            row_values.append(dp[c])

        all_values = " ".join(row_values)
        predict_value = predict(all_values)
        predictions.append(predict_value)

    return predictions

# uncomment for a quick test using data in test.csv
# print(predict_all("test.csv")) 
