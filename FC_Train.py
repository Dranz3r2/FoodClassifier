# IMPORTS
import numpy as np
import sklearn
import pandas
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



# All Helpers:

def clean_text(text):

  return text.lower().split()

def create_vocab(texts, max_words= None):
    word_counts = {}
    for text in texts:
        for word in clean_text(text):
            word_counts[word] = word_counts.get(word, 0) +1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    if max_words:
        sorted_words = sorted_words[:max_words]
    
    vocab = {}
    for i, word in enumerate(sorted_words):
        vocab[word[0]] = i

    return vocab

def make_vector(data, vocab):

  X = np.zeros([len(data), len(vocab)])
  t = np.zeros([len(data)])

  for i, dp in enumerate(data):
    if dp[1].lower() == "pizza":
      t[i] = 0
    elif dp[1].lower() == "shawarma":
      t[i] = 1
    elif dp[1].lower() == "sushi":
      t[i] = 2
    for word in vocab.keys():
      if word in clean_text(dp[0]):
        X[i][vocab[word]] += 1
  return X, t

def make_vector_label(data, vocab):

  t = np.zeros([len(data)])

  for i, dp in enumerate(data):
    if dp.lower() == "pizza":
      t[i] = 0
    elif dp.lower() == "shawarma":
      t[i] = 1
    elif dp.lower() == "sushi":
      t[i] = 2
  return t

def make_vector_text(data, vocab):
  X = np.zeros([len(data), len(vocab)])

  for i, dp in enumerate(data):

    for word in vocab.keys():

      if word in clean_text(dp):
        X[i][vocab[word]] += 1

  return X


# The neural network
class FoodClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.h1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.relu(self.h1(x))
        return self.h2(x)


# Training the model

df = pd.read_csv("cleaned_data_combined_modified.csv")
data_file = "cleaned_data_combined_modified.csv"

combined_qs = []
all_data = csv.reader(open(data_file))
for i, line in enumerate(all_data):
  if i != 0:
    combined_qs.append(
        [f'{line[1]} {line[2]} {line[3]} {line[4]} {line[5]} {line[6]} {line[7]} {line[8]}', f'{line[9]}']
    )

texts = []
labels = []
for dp in combined_qs:
    texts.append(item[0])
    labels.append(item[1])

X_train_texts, X_tv_texts, y_train_texts, y_tv_texts = train_test_split(texts, labels, test_size=0.2, random_state=42, shuffle=True)
X_val_texts, X_test_texts, y_val_texts, y_test_texts = train_test_split(X_tv_texts, y_tv_texts, test_size=0.5, random_state=42, shuffle=True)
vocab1 = create_vocab(X_train_texts, 700)

# vocab1 = create_vocab(X_train_texts, 500)

#TODO: Maybe add k fold cross valididation

X_train =make_vector_text(X_train_texts,vocab1)
y_train = make_vector_label(y_train_texts,vocab1)
X_val =make_vector_text(X_val_texts,vocab1)
y_val = make_vector_label(y_val_texts,vocab1)
X_test =make_vector_text(X_test_texts,vocab1)
y_test = make_vector_label(y_test_texts,vocab1)

# print(vocab1)
# print(X_val_texts)
# print(y_val_texts)


# change to tensors
X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val,dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset =TensorDataset(X_train_tensor, y_train_tensor)
train_loader =DataLoader(train_dataset,batch_size=32, shuffle=True)


model = FoodClassifier(input_dim=700, hidden_dim=64, output_dim=3)

# training using CEL
cel = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
num_epochs = 6

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cel(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)

    model.eval()
    with torch.no_grad():

        outputs = model(X_val_tensor)
        predicts = torch.argmax(outputs, dim=1)
        val_loss = cel(outputs, y_val_tensor)
        val_acc = (predicts == y_val_tensor).float().mean().item()
        val_accuracies.append(val_acc)
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}, Loss: {total_loss:.5f}, Validation Acc: {val_acc:.5f}", f"Val Loss: {val_loss.item():.5f}")


# the test
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicts = torch.argmax(outputs, dim=1)
    # print("predictions",predicts)
    # print(y_test_tensor)
    t_acc = (predicts == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {t_acc:.5f}")


#plot the graphs for training loss validation accuracy and validation loss
epochs = range(1, num_epochs+1)

plt.plot(epochs, train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(epochs, val_accuracies, color='orange')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(epochs, val_losses, color='red')
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


#Save the weights and vocab
np.save("h1_weights.npy",model.h1.weight.detach().numpy())
np.save("h1_bias.npy",model.h1.bias.detach().numpy())
np.save("h2_weights.npy",model.h2.weight.detach().numpy())
np.save("h2_bias.npy",model.h2.bias.detach().numpy())

with open("vocab.csv","w", newline="") as f:
    writer = csv.writer(f)
    for word, index in vocab1.items():
        writer.writerow([word, index])