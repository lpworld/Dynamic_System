import tensorflow as tf


class MF(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, 10])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_size])
        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        self.user_b = tf.get_variable('user_b', [user_count], initializer=tf.zeros_initializer())
        self.item_b = tf.get_variable('item_b', [item_count], initializer=tf.zeros_initializer())
        self.global_b = tf.get_variable('global_b', [], initializer=tf.zeros_initializer())

        user_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        item_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        u_b = tf.nn.embedding_lookup(self.user_b, self.u)
        i_b = tf.nn.embedding_lookup(self.item_b, self.i)

        self.logit = tf.reduce_sum(user_emb * item_emb, axis=1) + u_b + i_b + self.global_b
        self.score = tf.sigmoid(self.logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3],
            self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3], self.label: uij[5]})
        return uij[5], list(score), uij[0], uij[1]

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class BSAMF(object):
    # Bias-and-sensitivity-aware MF: clicks reweighted by inverse exposure propensity.
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, 10])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_size])
        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        self.user_b = tf.get_variable('user_b', [user_count], initializer=tf.zeros_initializer())
        self.item_b = tf.get_variable('item_b', [item_count], initializer=tf.zeros_initializer())
        self.global_b = tf.get_variable('global_b', [], initializer=tf.zeros_initializer())
        # learned per-item exposure logit
        self.item_exposure = tf.get_variable('item_exposure', [item_count],
                                             initializer=tf.zeros_initializer())

        user_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        item_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        u_b = tf.nn.embedding_lookup(self.user_b, self.u)
        i_b = tf.nn.embedding_lookup(self.item_b, self.i)

        self.logit = tf.reduce_sum(user_emb * item_emb, axis=1) + u_b + i_b + self.global_b
        self.score = tf.sigmoid(self.logit)

        prop = tf.sigmoid(tf.nn.embedding_lookup(self.item_exposure, self.i))
        weight = tf.stop_gradient(1.0 / (prop + 1e-6))
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit)
        self.loss = tf.reduce_mean(weight * bce)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3],
            self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3], self.label: uij[5]})
        return uij[5], list(score), uij[0], uij[1]

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class PropMF(object):
    # Propensity-scored MF (IPS), propensity clipped to [0.1, 1.0].
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, 10])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_size])
        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        self.user_b = tf.get_variable('user_b', [user_count], initializer=tf.zeros_initializer())
        self.item_b = tf.get_variable('item_b', [item_count], initializer=tf.zeros_initializer())
        self.global_b = tf.get_variable('global_b', [], initializer=tf.zeros_initializer())
        self.item_exposure = tf.get_variable('item_exposure', [item_count],
                                             initializer=tf.zeros_initializer())

        user_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        item_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        u_b = tf.nn.embedding_lookup(self.user_b, self.u)
        i_b = tf.nn.embedding_lookup(self.item_b, self.i)

        self.logit = tf.reduce_sum(user_emb * item_emb, axis=1) + u_b + i_b + self.global_b
        self.score = tf.sigmoid(self.logit)

        prop = tf.sigmoid(tf.nn.embedding_lookup(self.item_exposure, self.i))
        prop = tf.clip_by_value(prop, 0.1, 1.0)
        weight = tf.stop_gradient(1.0 / prop)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit)
        self.loss = tf.reduce_mean(weight * bce)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3],
            self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3], self.label: uij[5]})
        return uij[5], list(score), uij[0], uij[1]

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class DeepFM(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.last_i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, 10])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_size])
        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        # first-order weights
        self.user_w = tf.get_variable('user_w', [user_count], initializer=tf.zeros_initializer())
        self.item_w = tf.get_variable('item_w', [item_count], initializer=tf.zeros_initializer())
        self.global_b = tf.get_variable('global_b', [], initializer=tf.zeros_initializer())

        user_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        item_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        last_emb = tf.nn.embedding_lookup(self.item_emb_w, self.last_i)

        # first order
        first = (tf.nn.embedding_lookup(self.user_w, self.u)
                 + tf.nn.embedding_lookup(self.item_w, self.i)
                 + tf.nn.embedding_lookup(self.item_w, self.last_i)
                 + self.global_b)

        # second order over the three fields
        fields = tf.stack([user_emb, item_emb, last_emb], axis=1) # [B, 3, H]
        sum_sq = tf.square(tf.reduce_sum(fields, axis=1))
        sq_sum = tf.reduce_sum(tf.square(fields), axis=1)
        second = 0.5 * tf.reduce_sum(sum_sq - sq_sum, axis=1)

        fm_logit = first + second

        # DNN over concatenated embeddings
        concat = tf.concat([user_emb, item_emb, last_emb], axis=1)
        net = tf.layers.dense(concat, hidden_size, activation=tf.nn.relu, name='dnn1')
        net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu, name='dnn2')
        dnn_logit = tf.reshape(tf.layers.dense(net, 1, name='dnn_out'), [-1])

        self.logit = fm_logit + dnn_logit
        self.score = tf.sigmoid(self.logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0], self.i: uij[1], self.last_i: uij[2], self.hist: uij[3],
            self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
            self.u: uij[0], self.i: uij[1], self.last_i: uij[2],
            self.hist: uij[3], self.label: uij[5]})
        return uij[5], list(score), uij[0], uij[1]

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class DIN(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.u = tf.placeholder(tf.int32, [batch_size,])
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, 10])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])

        self.user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_size])
        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])

        user_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        cand = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist) # [B, T, H]

        # local activation unit: attention of each history item w.r.t. candidate
        cand_tile = tf.tile(tf.expand_dims(cand, 1), [1, 10, 1]) # [B, T, H]
        att_in = tf.concat([h_emb, cand_tile, h_emb * cand_tile, h_emb - cand_tile], axis=2)
        a = tf.layers.dense(att_in, hidden_size, activation=tf.nn.relu, name='att1')
        a = tf.layers.dense(a, 1, name='att2') # [B, T, 1]
        weights = tf.nn.softmax(tf.squeeze(a, axis=2)) # [B, T]
        pooled = tf.reduce_sum(h_emb * tf.expand_dims(weights, -1), axis=1) # [B, H]

        net = tf.concat([user_emb, cand, pooled], axis=1)
        net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu, name='fc1')
        net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu, name='fc2')
        self.logit = tf.reshape(tf.layers.dense(net, 1, name='fc_out'), [-1])
        self.score = tf.sigmoid(self.logit)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3],
            self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
            self.u: uij[0], self.i: uij[1], self.hist: uij[3], self.label: uij[5]})
        return uij[5], list(score), uij[0], uij[1]

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)
