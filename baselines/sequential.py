import tensorflow as tf
import numpy as np

MEMORY_WINDOW = 10


def layer_norm(x, name):
    # Per-feature normalization with learned scale and shift.
    with tf.variable_scope(name):
        d = x.shape.as_list()[-1]
        gamma = tf.get_variable('gamma', [d], initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', [d], initializer=tf.zeros_initializer())
        mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
        return gamma * (x - mean) / tf.sqrt(var + 1e-8) + beta


def multihead_attention(x, hidden_size, num_heads, mask, name):
    # Scaled dot-product self-attention. mask is [T, T] with -inf on disallowed
    # positions (causal) or all zeros (bidirectional).
    with tf.variable_scope(name):
        q = tf.layers.dense(x, hidden_size, name='q')
        k = tf.layers.dense(x, hidden_size, name='k')
        v = tf.layers.dense(x, hidden_size, name='v')
        q = tf.concat(tf.split(q, num_heads, axis=2), axis=0) # [B*h, T, d/h]
        k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)
        v = tf.concat(tf.split(v, num_heads, axis=2), axis=0)
        dk = hidden_size // num_heads
        logit = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / np.sqrt(dk)
        logit = logit + mask
        weight = tf.nn.softmax(logit)
        out = tf.matmul(weight, v) # [B*h, T, d/h]
        out = tf.concat(tf.split(out, num_heads, axis=0), axis=2) # [B, T, d]
        out = tf.layers.dense(out, hidden_size, name='proj')
    return out


def transformer_block(x, hidden_size, num_heads, mask, name):
    with tf.variable_scope(name):
        a = multihead_attention(x, hidden_size, num_heads, mask, 'attn')
        x = layer_norm(x + a, 'ln1')
        f = tf.layers.dense(x, hidden_size, activation=tf.nn.relu, name='ff1')
        f = tf.layers.dense(f, hidden_size, name='ff2')
        x = layer_norm(x + f, 'ln2')
    return x


class TransRec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.last_i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_size])
        item_b = tf.get_variable('item_b', [item_count],
                                 initializer=tf.zeros_initializer())

        user_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        last_emb = tf.nn.embedding_lookup(self.item_emb_w, self.last_i)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        bias = tf.nn.embedding_lookup(item_b, self.i)

        # user as a personalized translation from last item to next item
        trans = user_emb + last_emb - cand_emb
        sq_dist = tf.reduce_sum(tf.square(trans), axis=1)
        logit = bias - sq_dist
        self.score = tf.sigmoid(logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logit))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.last_i: uij[2],
                self.hist: uij[3], self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.last_i: uij[2],
                self.hist: uij[3], self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class SASRec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 num_blocks=2, num_heads=1, **kwargs):
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        pos_emb_w = tf.get_variable('pos_emb_w', [MEMORY_WINDOW, hidden_size])

        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        seq = h_emb + tf.expand_dims(pos_emb_w, 0)

        # causal mask: position t may not look ahead
        ones = tf.ones([MEMORY_WINDOW, MEMORY_WINDOW])
        lower = tf.matrix_band_part(ones, -1, 0)
        mask = (lower - 1.) * 1e9 # 0 where allowed, -inf above diagonal

        for b in range(num_blocks):
            seq = transformer_block(seq, hidden_size, num_heads, mask, 'block%d' % b)
        rep = seq[:, -1, :] # last position

        logit = tf.reduce_sum(rep * cand_emb, axis=1)
        self.score = tf.sigmoid(logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logit))
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


class BERT4Rec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 num_blocks=2, num_heads=1, **kwargs):
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        pos_emb_w = tf.get_variable('pos_emb_w', [MEMORY_WINDOW, hidden_size])

        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        seq = h_emb + tf.expand_dims(pos_emb_w, 0)

        mask = tf.zeros([MEMORY_WINDOW, MEMORY_WINDOW]) # bidirectional
        for b in range(num_blocks):
            seq = transformer_block(seq, hidden_size, num_heads, mask, 'block%d' % b)
        rep = tf.reduce_mean(seq, axis=1) # mean pool

        logit = tf.reduce_sum(rep * cand_emb, axis=1)
        self.score = tf.sigmoid(logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logit))
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


class Mamba4Rec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)

        # input-dependent selective recurrence h_t = a_t*h_{t-1} + b_t*x_t
        with tf.variable_scope('ssm'):
            state = tf.zeros([batch_size, hidden_size])
            for t in range(MEMORY_WINDOW):
                x = h_emb[:, t, :]
                a = tf.layers.dense(x, hidden_size, activation=tf.nn.sigmoid,
                                    name='gate_a', reuse=tf.AUTO_REUSE)
                b = tf.layers.dense(x, hidden_size, activation=tf.nn.sigmoid,
                                    name='gate_b', reuse=tf.AUTO_REUSE)
                state = a * state + b * x
        rep = state

        logit = tf.reduce_sum(rep * cand_emb, axis=1)
        self.score = tf.sigmoid(logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logit))
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
