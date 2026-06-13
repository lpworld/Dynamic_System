import numpy as np
import tensorflow as tf

MEMORY_WINDOW = 10


def _context(F, cand, hist):
    # x = normalize(F[cand] + mean over history rows)
    x = F[cand] + F[hist].mean(axis=0)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


class LinUCB(object):
    # Shared (disjoint-free) LinUCB over fixed random item features.
    def __init__(self, user_count, item_count, hidden_size, batch_size, alpha=0.25, **kwargs):
        self.H = hidden_size
        self.alpha = alpha
        rs = np.random.RandomState(625)
        self.F = rs.normal(size=[item_count, hidden_size]).astype(np.float64)
        self.A = np.eye(hidden_size)
        self.b = np.zeros(hidden_size)

    def train(self, sess, uij, lr):
        item, hist, click = uij[1], uij[3], uij[5]
        for k in range(len(item)):
            x = _context(self.F, item[k], hist[k])
            self.A += np.outer(x, x)
            self.b += click[k] * x
        return 0.0

    def test(self, sess, uij):
        item, hist = uij[1], uij[3]
        A_inv = np.linalg.inv(self.A)
        theta = A_inv.dot(self.b)
        scores = []
        for k in range(len(item)):
            x = _context(self.F, item[k], hist[k])
            ucb = self.alpha * np.sqrt(max(x.dot(A_inv).dot(x), 0.0))
            scores.append(float(theta.dot(x) + ucb))
        return list(uij[5]), scores, list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return self.F


class COFIBA(object):
    # Collaborative bandits: one LinUCB per user cluster (user_id % K).
    def __init__(self, user_count, item_count, hidden_size, batch_size, alpha=0.25, K=8, **kwargs):
        self.H = hidden_size
        self.alpha = alpha
        self.K = K
        rs = np.random.RandomState(625)
        self.F = rs.normal(size=[item_count, hidden_size]).astype(np.float64)
        self.A = [np.eye(hidden_size) for _ in range(K)]
        self.b = [np.zeros(hidden_size) for _ in range(K)]

    def train(self, sess, uij, lr):
        user, item, hist, click = uij[0], uij[1], uij[3], uij[5]
        for k in range(len(item)):
            c = user[k] % self.K
            x = _context(self.F, item[k], hist[k])
            self.A[c] += np.outer(x, x)
            self.b[c] += click[k] * x
        return 0.0

    def test(self, sess, uij):
        user, item, hist = uij[0], uij[1], uij[3]
        inv = [np.linalg.inv(a) for a in self.A]
        theta = [inv[c].dot(self.b[c]) for c in range(self.K)]
        scores = []
        for k in range(len(item)):
            c = user[k] % self.K
            x = _context(self.F, item[k], hist[k])
            ucb = self.alpha * np.sqrt(max(x.dot(inv[c]).dot(x), 0.0))
            scores.append(float(theta[c].dot(x) + ucb))
        return list(uij[5]), scores, list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return self.F


class REINFORCE(object):
    # Top-k REINFORCE: reward-weighted policy log-likelihood with a moving baseline.
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.reward = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.baseline = tf.placeholder(tf.float32, [])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)

        with tf.variable_scope('gru'):
            _, state = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(hidden_size),
                                         inputs=h_emb, dtype=tf.float32)

        pi = tf.reduce_sum(state * cand_emb, axis=1)
        self.score = tf.sigmoid(pi)
        adv = self.reward - self.baseline
        ll = self.reward * tf.log_sigmoid(pi) + (1 - self.reward) * tf.log_sigmoid(-pi)
        self.loss = -tf.reduce_mean(adv * ll)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self._base = 0.0

    def train(self, sess, uij, lr):
        r = np.mean(uij[5])
        self._base = 0.99 * self._base + 0.01 * r
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.reward: uij[5], self.baseline: self._base, self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class DRN(object):
    # Deep Q-network: one-step regression of Q to the immediate click reward.
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.reward = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        state = tf.reduce_mean(h_emb, axis=1)

        net = tf.concat([state, cand_emb], axis=1)
        net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu, name='q1')
        net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu, name='q2')
        q = tf.reshape(tf.layers.dense(net, 1, name='q_out'), [-1])
        self.score = q
        self.loss = tf.reduce_mean(tf.square(q - self.reward))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.reward: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class RLUR(object):
    # Actor-critic for user retention: actor over candidates, critic over state.
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.reward = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)

        with tf.variable_scope('gru'):
            _, state = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(hidden_size),
                                         inputs=h_emb, dtype=tf.float32)

        a = tf.reduce_sum(state * cand_emb, axis=1)
        self.score = tf.sigmoid(a)

        v = tf.layers.dense(state, hidden_size, activation=tf.nn.relu, name='v1')
        v = tf.reshape(tf.layers.dense(v, 1, name='v_out'), [-1])

        adv = self.reward - tf.stop_gradient(v)
        log_pi = self.reward * tf.log_sigmoid(a) + (1 - self.reward) * tf.log_sigmoid(-a)
        actor = -tf.reduce_mean(adv * log_pi)
        critic = tf.reduce_mean(tf.square(v - self.reward))
        self.loss = actor + critic
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.reward: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)
