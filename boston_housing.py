"""Implements Section 3.6 in Deep Learning with Python.

That section focuses on regression problems, by predicting
housing prices in Boston.
"""

from keras.datasets import boston_housing
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

def build_model(train_data, depth=3, width=64):
    model = models.Sequential()
    model.add(layers.Dense(width, activation='relu', input_shape=(train_data.shape[1],)))
    for _ in range(depth - 2):
        model.add(layers.Dense(width, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(), loss=losses.mse, metrics=[metrics.mae])
    return model

def prepare_data():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    # Following normalizes the input data
    train_mean = train_data.mean(axis=0)
    print(train_mean)
    train_data -= train_mean
    train_stddev = train_data.std(axis=0)
    train_data /= train_stddev
    test_data -= train_mean
    test_data /= train_stddev
    return train_data, train_targets, test_data, test_targets

EPOCHS = 100

def train_model(model, train_data, train_targets):
    """Train the model. 
    
    Use K-fold validation due to small number of training samples.
    """
    BUCKETS = 4    
    num_val_samples = len(train_data) // BUCKETS
    all_scores = []
    all_mae_histories = []
    for i in range(BUCKETS):
        print('Processing fold #', i)
        min_index = i * num_val_samples
        max_index = (i + 1) * num_val_samples
        val_data = train_data[min_index:max_index]
        val_targets = train_targets[min_index:max_index]
        partial_train_data = np.concatenate(
            [train_data[:min_index], train_data[max_index:]],
            axis = 0)
        partial_train_targets = np.concatenate(
            [train_targets[:min_index], train_targets[max_index:]],
            axis = 0)
        history = model.fit(partial_train_data, partial_train_targets, 
                            validation_data=(val_data, val_targets),
                            epochs=EPOCHS, batch_size=1, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)
    return all_scores, all_mae_histories

def plot_history(mae_history):
    average_mae_history = [
        np.mean([x[i] for x in mae_history]) for i in range(EPOCHS)]
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validate MAE')
    plt.show()

def main():
    train_data, train_targets, test_data, test_targets = prepare_data()
    model = build_model(train_data)
    all_scores, all_mae_histories = train_model(model, train_data, train_targets)
    print(all_scores)
    plot_history(all_mae_histories)
    
if __name__ == '__main__':
    main()