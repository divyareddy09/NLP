import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
# Data Preprocessing


# Define the CRF model
class CRFModel(tf.Module):
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.transitions = tf.Variable(initial_value=tf.random.uniform((num_labels, num_labels)))

    def forward(self, features):
        scores = tf.matmul(features, tf.transpose(self.transitions))
        return scores



# Instantiate the CRF model
num_labels = 9 
crf_model = CRFModel(num_labels)

def viterbi_decode(logits, transition_params):
    # Initialize viterbi variables
    viterbi = tf.TensorArray(dtype=tf.int32, size=logits.shape[0])
    backpointers = []

    # Forward pass
    for t in range(logits.shape[0]):
        if t == 0:
            viterbi_t = logits[t]
        else:
            viterbi_t = logits[t] + tf.reduce_max(viterbi_prev + transition_params, axis=1)
            backpointer = tf.argmax(viterbi_prev + transition_params, axis=1)
            backpointers.append(backpointer)
        viterbi_prev = viterbi_t
        viterbi = viterbi.write(t, viterbi_t)

    # Backtrack
    best_path = []

    # Find the best last tag
    best_last_tag = tf.argmax(viterbi_t)
    best_path.append(best_last_tag)

    for t in reversed(range(1, logits.shape[0])):
        best_last_tag = backpointers[t - 1][best_last_tag]
        best_path.append(best_last_tag)

    # Reverse the best path
    best_path.reverse()

    return best_path

def crf_loss(logits, labels, transition_params):
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params)
    loss = tf.reduce_mean(-log_likelihood)
    return loss

class CRFEstimator:
    def __init__(self, num_labels):
        self.model = CRFModel(num_labels)

    def fit(self, X, y):
        # X is your feature data, and y is your label data
        train_crf_model(self.model, X, y, num_epochs, learning_rate)
        return self

    def predict(self, X):
        logits = self.model.forward(X)
        return viterbi_decode(logits, self.model.transitions)

# Model Training
def train_crf_model(crf_model, features, labels, num_epochs, learning_rate):
    optimizer = tf.optimizers.Adam(learning_rate)
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = crf_model.forward(features)
            loss = crf_loss(logits, labels, crf_model.transitions)
        gradients = tape.gradient(loss, crf_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, crf_model.trainable_variables))

# Evaluate the CRF model
def evaluate_crf_model(crf_model, features, labels):
    logits = crf_model.forward(features)
    predicted_labels = viterbi_decode(logits, crf_model.transitions)
    # Calculate and print evaluation metrics

# Hyperparameter Tuning
param_grid = {
    'num_epochs': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1]
}

scorer = make_scorer(f1_score, average='weighted')  # Adjust the 'average' parameter as needed

# Use GridSearchCV with the custom estimator and scoring function
best_hyperparameters = GridSearchCV(estimator, param_grid, cv=3, scoring=scorer).fit(features, labels).best_params_

# Train the CRF model with the best hyperparameters
train_crf_model(estimator.model, features, labels, best_hyperparameters['num_epochs'], best_hyperparameters['learning_rate'])
