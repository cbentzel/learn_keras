"""Following Section 3.5 in 'Deep Learning with Python'."""

from keras.datasets import reuters
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
import numpy as np

def decode_to_text(word_index_array):
    """Decode an array of word indices to text."""
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    word_array = [reverse_word_index.get(i - 3, '?') for i in word_index_array]
    return ' '.join(word_array)

def vectorize_sequences(sequences, dimension=10000):
    """Convert a sequence of words to a binary presence vector."""
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension):
    """Convert a label to a one-hot vector."""
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def prepare_data():
    """Load and prepare data from reuters data set."""
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = to_one_hot(train_labels, 46)
    y_test = to_one_hot(test_labels, 46)
    return x_train, x_test, y_train, y_test

def build_model(depth=3, width=64, final_width=46):
    model = models.Sequential()
    model.add(layers.Dense(width, activation='relu', input_shape=(10000,)))
    for i in range(depth - 2):
        model.add(layers.Dense(width, activation='relu'))
    model.add(layers.Dense(final_width, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return model

def main():
    x_train, x_test, y_train, y_test = prepare_data()
    model = build_model()


if __name__ == '__main__':
    main()