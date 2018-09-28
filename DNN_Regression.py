from utils import *

"""Run this file"""

# Data handling
tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv("TestData1.csv", sep=",")
df.dropna(how='any', axis=0)
np.random.shuffle(df.values)
print(df.isnull().values.any())

training_examples, training_targets, \
    validation_examples, validation_targets, \
    test_examples, test_targets = data_preprocessor(df, 0.2, 0.8)

dnn_regressor, training_rmse, validation_rmse = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),  # 0.0007
    steps=5000,
    batch_size=70,
    hidden_units=[10, 5],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

plot_errors(training_rmse, validation_rmse)

test_root_mean_squared_error, r_square = predictor(dnn_regressor, test_examples, test_targets)

print("Final RMSE (on test data):   %0.4f" % test_root_mean_squared_error)
print("Final R^2 (on test data):   %0.4f" % r_square)
