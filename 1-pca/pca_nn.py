from common import *

################################################################################
# 3. Define neural network

# k as in PCA
hidden_units = 400
std_dev = 0.01

# initialize decoding weights to be transpose of encoding weights
w_encode_decode = tf.Variable(tf.random_normal([NUM_OF_FEATURES, hidden_units], stddev=std_dev), name='encode_weights')


# input is images and output is recontruction
x = tf.placeholder(tf.float32, shape=(None, NUM_OF_FEATURES), name='x')
y_true = x
y_pred = decode(x, w_encode_decode)

# cost funtion
alpha = 0.005
cost = tf.reduce_sum(tf.pow(y_true - y_pred, 2))
train_step = tf.train.AdamOptimizer(alpha).minimize(cost)

################################################################################
# 4. Start training

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

weights_before = sess.run(w_encode_decode)

iteration = 30000
batch_size = 3

costs = []

for iter in range(iteration):
    training_examples = np.random.permutation(400)[:batch_size]
    # _, c = sess.run([train_step, cost], feed_dict = {x: centered_faces_matrix[:, :10].T})
    _ = sess.run(train_step, feed_dict = {x: centered_faces_matrix[:,training_examples].T})
    
    if iter % 500 == 0:
        print(iter)
        print(sess.run(cost, feed_dict={x: centered_faces_matrix.T}))
    
################################################################################
# 5. Check result

weights = sess.run(w_encode_decode)
example_num = 80

example = faces_matrix[:, example_num]
plt.imshow(example.reshape((112, 92)), cmap='gray')
plt.show()

coefficients_before = np.dot(centered_faces_matrix[:, example_num], weights_before)
reconstruction_before = np.dot(weights, coefficients_before) + faces_mean[:, example_num]
plt.imshow(reconstruction_before.reshape((112, 92)), cmap='gray')
plt.show()

coefficients = np.dot(centered_faces_matrix[:, example_num], weights)
reconstruction = np.dot(weights, coefficients) + faces_mean[:, example_num]
plt.imshow(reconstruction.reshape((112, 92)), cmap='gray')
plt.show()