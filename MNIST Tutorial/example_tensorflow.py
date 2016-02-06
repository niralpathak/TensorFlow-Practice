from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# creating a placeholder variable capable of holding any number of MNIST images
# each flattened into a 784-dimensional vector
x = tf.placeholder(tf.float32, [None, 784])

# model parameters in machine learning applications = Variables in TensorFlow
# W and b are initialized to 0
W = tf.Variable(tf.zeros([784, 10])) # W has size 784x10 due to 10 classes
b = tf.Variable(tf.zeros([10])) # b is bias

# we reverse the matrix multiplicaton to deal with the "NONE" argument on line 8
y = tf.nn.softmax(tf.matmul(x, W) + b)


# training beginning using cross-entropy loss function

# creating a placeholder for 10 classes
y_ = tf.placeholder(tf.float32, [None, 10])

# implementing cross-entropy algorithm using TensorFlow methods
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# minimize the cross-entropy result using a learning rate of 0.01 with gradient 
# descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# run the training step 1000 times with a batch set of 100; this is implementing
# stochastic gradient descent
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)

  # feed in the placeholders defined above 
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# determine accuracy of our model
# tf.argmax(y,1) = label our model thinks
# tf.argmax(y_,1) = label that is actually correct
# gives us a list of booleans 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# cast list of booleans into floats
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# feed in placeholders and print out accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

