import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets.compat.v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Prints numpy arrays nicer
np.set_printoptions(precision=2, suppress=True, linewidth=100)

boston_housing = sklearn.datasets.load_boston()
# Uncomment the following line for a description of the dataset.
# print(boston_housing['DESCR'])
# Consider only the number-of-rooms feature for this experiment.
xs = boston_housing.data[:, list(boston_housing.feature_names).index('RM')]
ys = boston_housing.target

data = list(zip(xs, ys))

# Perform 60% / 40% training/test split
split_index = int(len(data) * 0.6)
train_data = data[:split_index]
test_data = data[split_index:]
print('Num training examples:', len(train_data))
print('Num testing examples:', len(test_data))

# Hyperparameters
learning_rate = 0.005
num_epochs = 100

# Model Definition
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

m = tf.Variable(1.0)
b = tf.Variable(0.0)

y_prediction = x * m + b

loss = (y - y_prediction) ** 2
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    time_before = time.time()
    losses = []  # Storing losses so we can plot them later
    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        cumulative_loss = 0
        for train_x, train_y in train_data:
            _loss, _train_op = sess.run(
                (loss, train_op), feed_dict={x: train_x, y: train_y})
            cumulative_loss += _loss
        average_loss = cumulative_loss / len(train_data)
        if epoch % 5 == 4:
            print('Epoch: {}, Loss: {}'.format(epoch + 1, average_loss))
        losses.append(average_loss)
    time_after = time.time()
    print('Training took {:.2f}s.'.format(time_after - time_before))

    # Introspection
    print()
    _m, _b = sess.run([m, b])
    print('Estimated m:', _m)
    print('Estimated b:', _b)

    # Prediction
    train_ys = []
    train_ys_prediction = []
    for train_x, train_y in train_data:
        train_ys.append(train_y)
        train_ys_prediction.append(
            sess.run(y_prediction, feed_dict={x: train_x}))
    train_ys = np.array(train_ys)
    train_ys_prediction = np.array(train_ys_prediction)

    test_ys = []
    test_ys_prediction = []
    for test_x, test_y in test_data:
        test_ys.append(test_y)
        test_ys_prediction.append(sess.run(y_prediction, feed_dict={x: test_x}))
    test_ys = np.array(test_ys)
    test_ys_prediction = np.array(test_ys_prediction)

plt.figure(dpi=150)
plt.title('Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(len(losses)), losses, color='#458588')
plt.show()

train_mean_squared_error = np.mean((train_ys - train_ys_prediction) ** 2)
test_mean_squared_error = np.mean((test_ys - test_ys_prediction) ** 2)

print('Mean Squared Error on Training data:', train_mean_squared_error)
print('Mean Squared Error on Testing data:', test_mean_squared_error)

plt.figure(dpi=150)
plt.title('Actual vs Predicted Data Points: Training Set')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.plot([min(train_ys), max(train_ys)], [min(train_ys), max(train_ys)],
         color='#1D2021', linestyle='--')
plt.scatter(train_ys, train_ys_prediction, color='#458588')
plt.show()

plt.figure(dpi=150)
plt.title('Actual vs Predicted Data Points: Testing Set')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.plot([min(test_ys), max(test_ys)], [min(test_ys), max(test_ys)],
         color='#1D2021', linestyle='--')
plt.scatter(test_ys, test_ys_prediction, color='#CC241D')
plt.show()