# Import necessary libraries
import numpy as np

# Loading training data
data_file_path = "/content/train.txt"

# Initialize an empty list to store the data
data = []

# Read the data from the file
with open(data_file_path, "r") as file:
    for line in file:
        data.append(line.strip())  # Assuming each line represents a sentence


# Initialize empty lists to store processed data
sentences = []  # List of sentences, each as a list of word tokens
labels = []     # List of labels, each as a list of NER labels

# Process the data
sentence = []
label = []
for line in data:
    if line.strip() == "-DOCSTART- -X- -X- O":
        # Skip document start lines
        continue
    if line == "":
        # Empty line indicates the end of a sentence
        sentences.append(sentence)
        labels.append(label)
        sentence = []
        label = []
    else:
        word, pos, np, ner = line.split()
        sentence.append(word)
        label.append(ner)

# Convert labels to numerical format (e.g., using a label-to-id mapping)
label_to_id = {
    "O": 0,
    "B-ORG": 1,
    "I-ORG": 2,
    "B-MISC": 3,
    "I-MISC": 4,
    "B-PER": 5,
    "I-PER": 6,
    "B-LOC": 7,
    "I-LOC": 8
}

labels = [[label_to_id[label] for label in sentence_labels] for sentence_labels in labels]

# We now have `sentences` as a list of word sequences and `labels` as a list of label sequences.
