# Initialize empty list to store features
features = []

# Define the feature window size (how many context words to consider)
window_size = 2

# Extract features for each sentence
for sentence in sentences:
    sentence_features = []
    sentence_length = len(sentence)
    for i in range(sentence_length):
        word = sentence[i]

        # Initialize feature vector for the current word
        feature_vector = []

        # Add features for the current word
        feature_vector.append(word)

        # Add features for previous words in the window
        for j in range(1, window_size + 1):
            if i - j >= 0:
                feature_vector.append(sentence[i - j])

        # Add features for next words in the window
        for j in range(1, window_size + 1):
            if i + j < sentence_length:
                feature_vector.append(sentence[i + j])

        # Append the feature vector to the sentence features
        sentence_features.append(feature_vector)

    features.append(sentence_features)

# Now, `features` is a list of feature sequences, where each feature sequence contains feature vectors for each word.
