import edward as ed
import tensorflow as tf
from edward.models import Beta, Bernoulli, Normal, Categorical, Empirical
from tensorflow.contrib import slim

## Beta-Binomial
theta = Beta(a=1.0, b=1.0)
x = Bernoulli(p=tf.ones(50) * theta)

## VAE for MNSIT
N = 1000
D = 28 * 28  # Data dimensions
d = D

##Probabilistic Model
z = Normal(mu=tf.zeros(N, d), sigma=tf.ones(N, d))
h = slim.fully_connected(z, 256, activation_fn=tf.nn.relu)
x = Bernoulli(logits=slim.fully_connected(h, 28 * 28, activation_fn=None))

##Variational Model
qx = tf.placeholder(tf.float32, [N, 28 * 28])
qh = slim.fully_connected(qx, 256, activation_fn=tf.nn.relu)
qz = z = Normal(mu=slim.fully_connected(qh, d, activation_fn=None),
                sigma=slim.fully_connected(qh, d, activation_fn=tf.nn.softplus))
## Bayesisan RNN

x_train = tf.placeholder(tf.float32, [None, D])
H = 128

by = Normal(mu=tf.zeros([1]), sigma=tf.ones([1]))
bh = Normal(mu=tf.zeros([H]), sigma=tf.ones([H]))
Wy = Normal(mu=tf.zeros([H, 1]), sigma=tf.ones([H, 1]))
Wx = Normal(mu=tf.zeros([D, H]), sigma=tf.ones([D, H]))
Wh = Normal(mu=tf.zeros([H, H]), sigma=tf.ones([H, H]))


def rnn_cell(hprev, xt):
    return tf.tanh(tf.dot(hprev, Wh) + tf.dot(xt, Wx) + bh)


h = tf.scan(rnn_cell, x, initializer=tf.zeros(H))
y = Normal(mu=tf.matmul(h, Wy) + by, sigma=1.0)

#### Hierarchichal
K = 5  # Number of clusters
beta = Normal(mu=tf.zeros[K, D], sigma=tf.ones([K, D]))
z = Categorical(logits=tf.zeros([N, K]))
x = Normal(mu=tf.gather(beta, z), sigma=tf.ones([N, D]))

# Model Variational Inference
qbeta = Normal(mu=tf.Variable(tf.zeros([K, D])), sigma=tf.exp(tf.Variable(tf.zeros([K, D]))))
qz = Categorical(logits=tf.Variable(tf.zeros([N, K])))
inference = ed.VariationalInference({beta: qbeta, z: qz}, data={x: x_train})
# Monte Carlo
T = 10000  # number of samples
qbeta = Empirical(params=tf.Variable(tf.zeros([T, K, D])))
qz = Empirical(params=tf.Variable(tf.zeros([T, N])))
inference = ed.MonteCarlo({beta: qbeta, z: qz}, data={x: x_train})

### GAN
M = 100


def generative_network(z):
    h = slim.fully_connected(z, 256, activation_fn=tf.nn.relu)
    return slim.fully_connected(h, 28 * 28, activation_fn=None)


def discriminative_network(x):
    h = slim.fully_connected(x, 28 * 28, activation_fn=tf.nn.relu)
    return slim.fully_connected(h, 1, activation_fn=None)


# Probabilist Model
z = Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
x = generative_network(z)

# Augumentation for GAN-based Inference
y_fake = Bernoulli(logits=discriminative_network(x))
y_real = Bernoulli(logits=discriminative_network(x_train))

data = {y_real: tf.ones(N), y_fake: tf.zeros(M)}
inference = ed.GANInference(data=data)
