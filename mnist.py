import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32,[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#----Weight Initialization---#
#One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
#Convolution and Pooling
#Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
#Our pooling is plain old max pooling over 2x2 blocks
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
 
x_image = tf.reshape(x, [-1,28,28,1])
 

#he convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32].
#The first two dimensions are the patch size,
#the next is the number of input channels, and the last is the number of output channels.
W_conv1 = weight_variable([5,5,1,32])

#We will also have a bias vector with a component for each output channel.
b_conv1 = bias_variable([32])

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
#The max_pool_2x2 method will reduce the image size to 14x14.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#----second convolution layer----#
#The second layer will have 64 features for each 5x5 patch and input size 32.
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#----fully connected layer----#
#Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#-----dropout------#
#To reduce overfitting, we will apply dropout before the readout layer.
#We create a placeholder for the probability that a neuron's output is kept during dropout.
#This allows us to turn dropout on during training, and turn it off during testing.
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

#------train and evaluate----#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0],
                                                       y_: batch[1],
                                                       keep_prob: 1.})
            print('setp {},the train accuracy: {}'.format(i, train_accuracy))
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    test_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.})
    print('the test accuracy :{}'.format(test_accuracy))
    saver = tf.train.Saver()
    path = saver.save(sess, './my_net/mnist_deep.ckpt')
    print('save path: {}'.format(path))