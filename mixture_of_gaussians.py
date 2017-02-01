import edward as ed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Categorical, InverseGamma, Mixture, MultivariateNormalDiag, Normal, Dirichlet

plt.style.use('ggplot')


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    stds = [[0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


# # MODEL Version One with marginalized mixture assignments

def build_marginalized_model():
    mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
    sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
    cat = Categorical(logits=tf.zeros([N, K]))
    components = [
        MultivariateNormalDiag(mu=tf.ones([N, 1]) * tf.gather(mu, k),
                               diag_stdev=tf.ones([N, 1]) * tf.gather(sigma, k))
        for k in range(K)]

    x = Mixture(cat=cat, components=components)
    print("mu: {}, sigma: {}, cat: {}, components: {}, x: {}".format(mu, sigma, cat, components, x))
    return x


def build_dirichlet_model():
    pi = Dirichlet(alpha=tf.ones(K))
    mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
    sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
    cat = Categorical(logits=tf.ones([N, 1]) * ed.logit(pi))
    x = Normal(mu=tf.gather(mu, cat), sigma=tf.gather(sigma, c))
    print(tf.gather(mu, cat))

    print("mu: {}, sigma: {}, cat: {}, x: {}".format(mu, sigma, cat, x))
    return x


N = 500  # number of data points
K = 2  # number of components
D = 2  # dimensionality of data

ed.set_seed(42)

# DATA
x_train = build_toy_dataset(N)
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.title("Simulated dataset")
plt.show()

mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
cat = Categorical(logits=tf.zeros([N, K]))
components = [
    MultivariateNormalDiag(mu=tf.ones([N, 1]) * tf.gather(mu, k),
                           diag_stdev=tf.ones([N, 1]) * tf.gather(sigma, k))
    for k in range(K)]

x = Mixture(cat=cat, components=components)
print("mu: {}, sigma: {}, cat: {}, components: {}, x: {}".format(mu, sigma, cat, components, x))

# INFERENCE
qmu = Normal(
    mu=tf.Variable(tf.random_normal([K, D])),
    sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
qsigma = InverseGamma(
    alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
    beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: x_train})
inference.initialize(n_samples=20, n_iter=4000)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']
    if t % inference.n_print == 0:
        print("Inferred cluster means:")
        print(sess.run(qmu.value()))

# Average per-cluster and per-data point likelihood over many posterior samples.
log_liks = []
for _ in range(100):
    mu_sample = qmu.sample()
    sigma_sample = qsigma.sample()
    # Take per-cluster and per-data point likelihood.
    log_lik = []
    for k in range(K):
        # Scalar indices tf.gather(mu_sample, k),
        # output[:, ..., :] = params[indices, :, ...:]
        x_post = Normal(mu=tf.ones([N, 1]) * tf.gather(mu_sample, k),
                        sigma=tf.ones([N, 1]) * tf.gather(sigma_sample, k))
        log_lik.append(tf.reduce_sum(x_post.log_prob(x_train), 1))

    log_lik = tf.stack(log_lik)  # has shape (K, N)
    log_liks.append(log_lik)

log_liks = tf.reduce_mean(log_liks, 0)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 0).eval()
plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()
