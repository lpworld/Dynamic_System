import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

MEMORY_WINDOW = 10


def encode_state(item_emb_w, hist, hidden_size):
    # GRU over the history, last hidden state is the user state.
    h_emb = tf.nn.embedding_lookup(item_emb_w, hist)
    with tf.variable_scope('state', reuse=tf.AUTO_REUSE):
        _, state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
    return state


class PrefRec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        state = encode_state(self.item_emb_w, self.hist, hidden_size)

        # reward model r(state, cand)
        x = tf.concat([state, cand_emb], axis=1)
        x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu, name='r1')
        reward = tf.squeeze(tf.layers.dense(x, 1, name='r2'), axis=1) # [B]
        self.score = reward

        # pairwise preference: clicked candidates ranked above non-clicked ones.
        # compare each reward against the mean reward of the opposite label.
        pos = self.label
        neg = 1. - self.label
        n_pos = tf.reduce_sum(pos) + 1e-8
        n_neg = tf.reduce_sum(neg) + 1e-8
        mean_pos = tf.reduce_sum(reward * pos) / n_pos
        mean_neg = tf.reduce_sum(reward * neg) / n_neg
        diff = pos * (reward - mean_neg) + neg * (mean_pos - reward)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(diff))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class DeepExploration(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 num_heads=5, beta=0.5, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        state = encode_state(self.item_emb_w, self.hist, hidden_size)

        # shared trunk, K bootstrapped Q-heads
        trunk = tf.concat([state, cand_emb], axis=1)
        trunk = tf.layers.dense(trunk, hidden_size, activation=tf.nn.relu, name='trunk')

        qs = []
        loss = 0.
        for k in range(num_heads):
            q = tf.squeeze(tf.layers.dense(trunk, 1, name='head%d' % k), axis=1)
            qs.append(q)
            mask = tf.cast(tf.random_uniform([batch_size]) < 0.5, tf.float32)
            loss += tf.reduce_mean(mask * tf.square(q - self.label))
        self.loss = loss

        q_stack = tf.stack(qs, axis=1) # [B, K]
        mean, var = tf.nn.moments(q_stack, axes=[1])
        self.score = mean + beta * tf.sqrt(var + 1e-8) # uncertainty bonus
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class HAC(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 sup_weight=0.5, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        state = encode_state(self.item_emb_w, self.hist, hidden_size)

        # actor produces a latent action z in item space
        a = tf.layers.dense(state, hidden_size, activation=tf.nn.relu, name='actor1')
        z = tf.layers.dense(a, hidden_size, name='actor2') # [B, H]

        # recommend by proximity in latent action space
        self.score = -tf.reduce_sum(tf.square(z - cand_emb), axis=1)

        # critic estimates value of (state, z)
        def critic(s, act):
            c = tf.concat([s, act], axis=1)
            c = tf.layers.dense(c, hidden_size, activation=tf.nn.relu,
                                name='critic1', reuse=tf.AUTO_REUSE)
            return tf.squeeze(tf.layers.dense(c, 1, name='critic2',
                                              reuse=tf.AUTO_REUSE), axis=1)

        value = critic(state, tf.stop_gradient(z))
        critic_loss = tf.reduce_mean(tf.square(value - self.label))
        actor_value = critic(state, z)
        actor_loss = -tf.reduce_mean(actor_value)
        # supervised pull of z toward clicked candidate embeddings
        sup = self.label * tf.reduce_sum(tf.square(z - cand_emb), axis=1)
        sup_loss = tf.reduce_sum(sup) / (tf.reduce_sum(self.label) + 1e-8)

        self.loss = critic_loss + actor_loss + sup_weight * sup_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class ResAct(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 rec_weight=1.0, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        state = encode_state(self.item_emb_w, self.hist, hidden_size)

        # reconstruct a target action embedding from state
        t = tf.layers.dense(state, hidden_size, activation=tf.nn.relu, name='rec1')
        t = tf.layers.dense(t, hidden_size, name='rec2') # [B, H]
        # reconstruction pulls t toward clicked candidates, weighted by reward
        rec = self.label * tf.reduce_sum(tf.square(t - cand_emb), axis=1)
        rec_loss = tf.reduce_sum(rec) / (tf.reduce_sum(self.label) + 1e-8)

        # residual refinement
        res = tf.layers.dense(state, hidden_size, activation=tf.nn.relu, name='res1')
        res = tf.layers.dense(res, hidden_size, name='res2')
        r = t + res

        logit = tf.reduce_sum(r * cand_emb, axis=1)
        self.score = tf.sigmoid(logit)
        bce = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logit))
        self.loss = rec_weight * rec_loss + bce
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)
