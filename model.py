import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
import numpy as np


def spectral_norm(w, iteration=1):
    # Power-iteration estimate of the largest singular value (Miyato et al., 2018).
    w_shape = w.shape.as_list()
    w_mat = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(),
                        trainable=False)
    u_hat = u
    v_hat = None
    for _ in range(iteration):
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w_mat)), dim=1)
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_mat), dim=1)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_mat / sigma, w_shape)
    return w_norm


def dense_sn(x, units, activation=None, name=None):
    # Dense layer whose kernel is divided by its spectral norm, bounding the
    # Lipschitz constant of the evolution function by construction.
    with tf.variable_scope(name):
        w = tf.get_variable('kernel', [x.shape.as_list()[-1], units])
        b = tf.get_variable('bias', [units], initializer=tf.zeros_initializer())
        y = tf.matmul(x, spectral_norm(w)) + b
        if activation is not None:
            y = activation(y)
    return y


class FDSR(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 mdn_components=4, branch_num=3, spectral_normalize=True,
                 mdn_weight=0.1, margin=0.1):
        memory_window = 10
        self.margin = margin
        self.mdn_components = mdn_components
        self.branch_num = branch_num

        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.last_i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.next_hist = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.label = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.lr = tf.placeholder(tf.float64, [])
        self.i_table = tf.placeholder(tf.int32, [item_count,]) # [I]

        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size])
        user_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size])
        item_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
        last_item_emb = tf.nn.embedding_lookup(item_emb_w, self.last_i)
        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist)
        next_h_emb = tf.nn.embedding_lookup(item_emb_w, self.next_hist)
        item_emb_table = tf.nn.embedding_lookup(item_emb_w, self.i_table)

        ### Part I: Evolution Function f(X,t)
        # User Feature Embedding \theta_u (Self-Attentive GRU)
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
            feature_emb = self.seq_attention(output, h_emb)
            next_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=next_h_emb, dtype=tf.float32)
            next_feature_emb = self.seq_attention(next_output, next_h_emb)

        # Concatenation of \theta_u, user & item embedding (MLP Layer, Lipschitz-controlled)
        dense = dense_sn if spectral_normalize else (
            lambda x, units, activation=None, name=None:
            tf.layers.dense(x, units, activation=activation, name=name))
        evolution = tf.concat([user_emb, last_item_emb, feature_emb], axis=1)
        evolution = dense(evolution, hidden_size, activation=tf.nn.sigmoid, name='evolution')

        # Expectation of Next Evolution State (Mixture Density Network)
        # The MDN reads the current state and outputs a Gaussian mixture over the
        # next trajectory point; the forward-looking term E[X_next] is the closed-form
        # mixture mean \sum_i alpha_i mu_i, so no sampling is required.
        state = tf.concat([user_emb, last_item_emb, feature_emb], axis=1)
        alpha, mu, log_sigma = self.mdn_head(state, hidden_size)
        expectation_next_evolution = tf.reduce_sum(
            tf.expand_dims(alpha, -1) * mu, axis=1) # [B, H], \sum_i alpha_i mu_i

        # The Complete Evolution Function (with Forward-Looking)
        function = evolution + expectation_next_evolution

        ### Part II: Solving the Dynamic System (DeepONet)
        # Branch Networks
        branch_output = tf.concat(
            [tf.layers.dense(function, hidden_size, activation=tf.nn.sigmoid, name='branch%d' % k)
             for k in range(self.branch_num)], axis=1) # [B, branch_num*H]

        # Trunk Network
        trunk_output = tf.layers.dense(last_item_emb, self.branch_num*hidden_size, activation=tf.nn.sigmoid)

        # Aggregation for Obtaining Numeric Solution
        target_item_emb = tf.multiply(branch_output, trunk_output)
        target_item_emb = tf.layers.dense(target_item_emb, hidden_size, activation=tf.nn.sigmoid)

        ### Part III: Generate Product Recommendations
        distance = tf.norm(target_item_emb-item_emb, ord='euclidean', axis=1)
        self.distance = distance
        # Squared-hinge contrastive loss (Eq. 7): positives pulled in, negatives
        # pushed beyond the margin. Mixture-density negative log-likelihood of the
        # observed next interaction is added as a joint training signal.
        contrastive = self.label * tf.square(distance) + \
            (1 - self.label) * tf.square(tf.maximum(0., self.margin - distance))
        mdn_nll = self.mdn_nll(alpha, mu, log_sigma, tf.stop_gradient(next_feature_emb))
        self.loss = 0.5 * tf.reduce_mean(contrastive) + mdn_weight * tf.reduce_mean(mdn_nll)

        ### Step Variables for Back-Propogation
        self.embedding = target_item_emb
        self.item_emb_table = item_emb_table
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.last_i: uij[2],
                self.hist: uij[3],
                self.next_hist: uij[4],
                self.label: uij[5],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        distance, embedding = sess.run([self.distance, self.embedding], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.last_i: uij[2],
                self.hist: uij[3],
                self.next_hist: uij[4],
                self.label: uij[5]
                })
        return uij[5], list(distance), uij[0], uij[1], embedding

    def get_item_emb(self, sess, i_table):
        item_emb_table = sess.run(self.item_emb_table, feed_dict={
                self.i_table: i_table
                })
        return item_emb_table

    def mdn_head(self, state, hidden_size):
        # One forward pass maps the current state to the parameters of an
        # m-component Gaussian mixture over the next trajectory point.
        m = self.mdn_components
        h = tf.layers.dense(state, hidden_size, activation=tf.nn.tanh, name='mdn_hidden')
        alpha = tf.nn.softmax(tf.layers.dense(h, m, name='mdn_alpha')) # [B, m]
        mu = tf.layers.dense(h, m*hidden_size, name='mdn_mu')
        mu = tf.reshape(mu, [-1, m, hidden_size]) # [B, m, H]
        log_sigma = tf.layers.dense(h, m, name='mdn_log_sigma') # [B, m], isotropic
        return alpha, mu, log_sigma

    def mdn_nll(self, alpha, mu, log_sigma, target):
        # Negative log-likelihood of the observed next state under the mixture
        # (isotropic Gaussian components, shared variance per component).
        hidden_size = mu.shape.as_list()[-1]
        target = tf.expand_dims(target, 1) # [B, 1, H]
        sigma = tf.exp(log_sigma) # [B, m]
        sq = tf.reduce_sum(tf.square(target - mu), axis=2) # [B, m]
        log_comp = (-0.5 * sq / tf.square(sigma)
                    - hidden_size * log_sigma
                    - 0.5 * hidden_size * np.log(2 * np.pi)) # [B, m]
        log_mix = tf.reduce_logsumexp(tf.log(alpha + 1e-12) + log_comp, axis=1)
        return -log_mix

    def seq_attention(self, gru_output, item_seq):
        # Euclidean attention: weight each step by how far the user's interest
        # moved at that step, i.e. the distance between consecutive item
        # embeddings, so that preference shifts receive higher weight.
        shift = item_seq - tf.concat([item_seq[:, :1, :], item_seq[:, :-1, :]], axis=1)
        score = tf.norm(shift, ord='euclidean', axis=2) # [B, T]
        alphas = tf.nn.softmax(score)
        output = tf.reduce_sum(gru_output * tf.expand_dims(alphas, -1), axis=1)
        return output
