import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Prints numpy arrays nicer
np.set_printoptions(precision=2, suppress=True, linewidth=100)

def target_function(x):
    return x * 0.5 - 4


num_samples = 30
# Randomly sampled values in [-10, 10]
xs = np.random.uniform(low=-10, high=10, size=num_samples)
# Intended target value plus random noise
ys = target_function(xs) + np.random.normal(loc=0, scale=1, size=num_samples)

data = np.array(list(zip(xs, ys)))
print('data:')
print(data)

plt.figure(dpi=150)
plt.title('Data')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([-12, 12], [target_function(-12), target_function(12)],
         color='#458588', label='target_function')
plt.scatter(xs, ys, color='#458588', label='data')
plt.legend()
plt.savefig("mygraph.png")
# plt.show(block=True)
# plt.pause(15)
# plt.close()

# Hyperparameters
learning_rate = 0.005
num_epochs = 20

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
    losses = []  # Storing losses so we can plot them later
    for epoch in range(num_epochs):
        np.random.shuffle(data)
        cumulative_loss = 0
        for _x, _y in data:
            _loss, _train_op = sess.run(
                (loss, train_op), feed_dict={x: _x, y: _y})
            cumulative_loss += _loss
        average_loss = cumulative_loss / len(data)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, average_loss))
        losses.append(average_loss)

    # Introspection
    print()
    _m, _b = sess.run([m, b])
    print('Estimated m:', _m)
    print('Estimated b:', _b)

    # Prediction
    ys_actual = []
    ys_predicted = []
    for _x, _y in data:
        ys_actual.append(_y)
        ys_predicted.append(sess.run(y_prediction, feed_dict={x: _x}))

plt.figure(dpi=150)
plt.title('Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(len(losses)), losses, color='#458588')
plt.show()
plt.savefig("mygraph2.png")


plt.figure(dpi=150)
plt.title('Actual Function vs Estimated Function')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([-12, 12], [target_function(-12), target_function(12)],
         color='#458588', label='target_function')
plt.scatter(xs, ys, color='#458588', label='data')
plt.plot([-12, 12], [-12 * _m + _b, 12 * _m + _b],
         color='#CC241D', label='estimated_function')
plt.legend()
plt.show()
plt.savefig("mygraph3.png")

plt.figure(dpi=150)
plt.title('Actual vs Predicted Data Points')
plt.xlabel('Actual y-value')
plt.ylabel('Predicted y-value')
plt.plot([min(ys_actual), max(ys_actual)], [min(ys_actual), max(ys_actual)],
         color='#1D2021', linestyle='--')
plt.scatter(ys_actual, ys_predicted, color='#458588')
plt.show()
plt.savefig("mygraph4.png")