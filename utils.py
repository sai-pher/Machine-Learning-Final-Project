import math
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

selected_features_data = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'tempo', 'valence',
    '0',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
    '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
    '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
    '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
    '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
    '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
    '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
    '121', '122', '123', '124', '125', '126', '127', '128', '129', '130',
    '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
    '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
    '151', '152', '153', '154', '155', '156', '157', '158', '159', '160',
    '161', '162', '163', '164', '165', '166', '167', '168', '169', '170',
    '171', '172', '173', '174', '175', '176', '177', '178', '179', '180',
    '181', '182', '183', '184', '185', '186', '187', '188', '189', '190',
    '191', '192', '193', '194', '195', '196', '197', '198', '199', '200',
    '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
    '211', '212', '213', '214', '215', '216', '217', '218', '219', '220',
    '221', '222', '223']


def data_cutter(df, test_prop, train_prop):
    """Preprocesses the data frame, separating it into the various data sets to be used by the training graph.

        :param df: A Pandas DataFrame expected to contain data from the spotify data set.
        :param test_prop: The proportion of the data expected to be used for testing the prediction as a decimal between 0 and 1.
            This set is never used in the training ad validation.
        :param train_prop: The proportion of the data expected to be used for training as a decimal between 0 and 1.
        :return: tuple of (train_set, valid_set, test_set)

    """

    # data size handling
    size = df.shape[0]
    test = int(size * test_prop)
    train = int(size * train_prop)
    valid = int(size * (1 - train_prop))

    # test data partitioning
    test_set = df.tail(test)

    # training and validation data partitioning
    dat = df.head(int(size - size * test_prop))
    train_set = dat.head(train)
    valid_set = dat.tail(valid)

    # clear used variables
    del test
    del train
    del valid
    del dat

    return train_set, valid_set, test_set


# google
def preprocess_features(df):
    """
    Prepares input features from audio dataset.

    :param df: A Pandas DataFrame expected to contain data from the Spotify audio data set.
    :return: processed_features

    """
    selected_features = df[selected_features_data]
    processed_features = selected_features.copy()
    return processed_features


# google
def preprocess_targets(df):
    """
    Prepares target features (i.e., labels) from spotify audio data set.

        :param df: A Pandas DataFrame expected to contain data from the spotify data set.
        :return: A DataFrame that contains the target feature.
    """

    output_targets = pd.DataFrame()
    output_targets["song_hotttnesss"] = df["song_hotttnesss"]
    return output_targets


# google
def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    :param input_features: The names of the numerical input features to use.
    :return: A set of feature columns
  """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


# google
def input_fn(features, targets, batch_size=1, shuffle=True, shuffle_size=10000, num_epochs=None):
    """Trains a neural network model.

      :param features: pandas DataFrame of features
      :param targets: pandas DataFrame of targets
      :param batch_size: Size of batches to be passed to the model
      :param shuffle: True or False. Whether to shuffle the data.
      :param shuffle_size: The size of the shuffle buffer. This defaults to 10,000
      :param num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
      :return: Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(shuffle_size)  # defaults to 10000

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def data_preprocessor(df, test_proportion=0.2, data_proportion=0.8):
    """Preprocesses the data frame, separating it into the various data sets to be used by the training graph.

        :param df: A Pandas DataFrame expected to contain data from the spotify data set.
        :param test_proportion: The proportion of the data expected to be used for testing the prediction as a decimal
            between 0 and 1. This set is never used in the training ad validation.
        :param data_proportion: The proportion of the data expected to be used for training as a decimal between 0 and 1.
        :return: tuple of (training_examples, training_targets, validation_examples, validation_targets, test_examples,
            test_targets)

    """
    train_set, valid_set, test_set = data_cutter(df, test_proportion, data_proportion)

    # examples for training.
    training_examples = preprocess_features(train_set)
    training_targets = preprocess_targets(train_set)

    # examples for validation.
    validation_examples = preprocess_features(valid_set)
    validation_targets = preprocess_targets(valid_set)

    # examples for validation.
    test_examples = preprocess_features(test_set)
    test_targets = preprocess_targets(test_set)

    return training_examples, training_targets, validation_examples, validation_targets, test_examples, test_targets

def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network regression model.

  In addition to training, this function also prints training progress information.

      :param my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      :param steps:  A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      :param batch_size: A non-zero `int`, the batch size.
      :param hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      :param training_examples: A `DataFrame` containing one or more columns from
        `df` to use as input features for training.
      :param training_targets: A `DataFrame` containing exactly one column from
        `df` to use as target for training.
      :param validation_examples: A `DataFrame` containing one or more columns from
        `df` to use as input features for validation.
      :param validation_targets: A `DataFrame` containing exactly one column from
        `df` to use as target for validation.
      :return: A tuple `(estimator, training_losses, validation_losses)`:
        estimator: the trained `DNNRegressor` object.
        training_losses: a `list` containing the training loss values taken during training.
        validation_losses: a `list` containing the validation loss values taken during training.
  """

    start_time = datetime.now()

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer)

    # Create input functions.

    training_input_fn = lambda: input_fn(training_examples,
                                         training_targets["song_hotttnesss"],
                                         batch_size=batch_size, shuffle_size=training_examples.shape[0])
    predict_training_input_fn = lambda: input_fn(training_examples,
                                                 training_targets["song_hotttnesss"],
                                                 num_epochs=1,
                                                 shuffle=False)
    predict_validation_input_fn = lambda: input_fn(validation_examples,
                                                   validation_targets["song_hotttnesss"],
                                                   num_epochs=1,
                                                   shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period)

        # compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))

        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # print the current loss.
        print("  period %02d : %0.4f" % (period, training_root_mean_squared_error))

        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished at step %d." % steps_per_period * period)

    print("Model description: Steps: %d || Architecture: %s || Buffer size: %d "
          % (steps, hidden_units, training_examples.shape[0]))

    print("Final RMSE (on training data):   %0.4f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.4f" % validation_root_mean_squared_error)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    return dnn_regressor, training_rmse, validation_rmse


def plot_errors(training_rmse, validation_rmse):
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()


def predictor(dnn_regressor, test_examples, test_targets):
    """

    :param dnn_regressor: The DNNRegressor model to be used in the prediction.
    :param test_examples: A `DataFrame` containing one or more columns from
      `df` to use as input features for prediction.
    :param test_targets: A `DataFrame` containing exactly one column from
      `df` to use as target for prediction.
    :return: test_root_mean_squared_error, r_square
    """
    predict_test_input_fn = lambda: input_fn(test_examples,
                                             test_targets["song_hotttnesss"],
                                             num_epochs=1,
                                             shuffle=False)
    test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    test_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    r_square = sklearn.metrics.r2_score(test_targets, test_predictions)

    return test_root_mean_squared_error, r_square
