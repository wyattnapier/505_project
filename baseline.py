import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier

# for plotting results
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

file_name = "./DepressionEmo-main/Dataset/train.json"
file_path = os.path.join(os.getcwd(), file_name)

label_file_name="./DepressionEmo-main/Dataset/train.json" # had to manually remove 'unlabelled' to convert these to bitmask
label_file_path = os.path.join(os.getcwd(), label_file_name)

df = pd.read_json(file_path, lines=True)
df = df[df['label_id'] != 'unlabelled'] # drop unlabelled rows

df['text'] = df['text'].str.lower() # turn to lower

max_bits = max(df['label_id']).bit_length() # get max len

# Function to convert integer labels to binary vector
def int_to_bitmask(label, num_bits):
    binary_rep = bin(label)[2:]  # Convert to binary string without '0b' prefix
    return [int(bit) for bit in binary_rep.zfill(num_bits)]  # Pad left with zeros

# Convert all labels
y = np.array([int_to_bitmask(label, max_bits) for label in df['label_id']])

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y, test_size=0.2, random_state=0
)

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

nb_clf = OneVsRestClassifier(BernoulliNB())
nb_clf.fit(X_train_bow, y_train)

nb_pred = nb_clf.predict(X_test_bow)
# f1 score is more representative since accuracy only counts if all bits match
# nb_accuracy = accuracy_score(y_test, nb_pred)
# print(f"Naive Bayes Classifier Accuracy: {nb_accuracy:.4f}")

print("\nClassification Report for Naive Bayes Classifier:")
print(classification_report(y_test, nb_pred))