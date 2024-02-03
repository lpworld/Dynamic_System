import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
import scipy.spatial
from collections import Counter
import numpy as np

class Dynamic_System(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size):
        memory_window = 10
        self.margin = 0.1
        
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

        mdn_weight_w = tf.get_variable("mdn_weight_w", [item_count, hidden_size])
        mdn_weight = tf.nn.embedding_lookup(user_emb_w, self.last_i)
        mdn_bias_w = tf.get_variable("mdn_bias_w", [item_count, hidden_size])
        item_emb = tf.nn.embedding_lookup(item_emb_w, self.last_i)

        ### Part I: Evolution Function f(X,t)
        # User Feature Embedding \theta_u
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
        	output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
        	feature_emb, _ = self.seq_attention(output, hidden_size, memory_window)
        	next_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=next_h_emb, dtype=tf.float32)
        	next_feature_emb, _ = self.seq_attention(next_output, hidden_size, memory_window)

        # Concatenation of \theta_u, user & item embedding (MLP Layer)
        evolution = tf.concat([user_emb, last_item_emb, feature_emb], axis=1)
        evolution = tf.layers.dense(evolution, hidden_size, activation=tf.nn.sigmoid)

        # Expectation of Next Evolution State (Mixture Density Network)
        next_evolution = tf.concat([user_emb, item_emb, next_feature_emb], axis=1)
        next_evolution = tf.layers.dense(next_evolution, hidden_size, activation=tf.nn.sigmoid)
        expectation_next_evolution = tf.layers.dense(next_evolution, hidden_size, activation=tf.nn.sigmoid)

        # The Complete Evolution Function (with Forward-Looking)
        function = evolution + expectation_next_evolution

        ### Part II: Solving the Dynamic System (DeepONet)
        #Branch Networks
        self.branch_num = 3
        branch_output1 = tf.layers.dense(function, hidden_size, activation=tf.nn.sigmoid, name='branch1')
        branch_output2 = tf.layers.dense(function, hidden_size, activation=tf.nn.sigmoid, name='branch2')
        branch_output3 = tf.layers.dense(function, hidden_size, activation=tf.nn.sigmoid, name='branch3')
        branch_output = tf.concat([branch_output1,branch_output2,branch_output3], axis=1) #B*3*H

        # Trunk Network
        trunk_output = tf.layers.dense(last_item_emb, self.branch_num*hidden_size, activation=tf.nn.sigmoid) #B*3

        # Aggregation for Obtaining Numeric Solution
        target_item_emb = tf.multiply(branch_output, trunk_output)
        target_item_emb = tf.layers.dense(target_item_emb, hidden_size, activation=tf.nn.sigmoid)

        ### Part III: Generate Product Recommendations
        distance = tf.norm(target_item_emb-item_emb, ord='euclidean', axis=1)
        self.distance = distance
        self.loss = 0.5 * tf.reduce_mean(self.label * tf.square(tf.maximum(0., self.margin - distance)) + (1 - self.label) * tf.square(distance))

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
        distance = [float(x < self.margin) for x in distance]
        return uij[5], distance, uij[0], uij[1], embedding

    def get_item_emb(self, sess, i_table):
        item_emb_table = sess.run(self.item_emb_table, feed_dict={
                self.i_table: i_table
                })
        return item_emb_table
    
    def seq_attention(self, inputs, hidden_size, attention_size):
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        records = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        user, item, last_item, hist, next_hist, click = [], [], [], [], [], []
        for record in records:
            user.append(record[0])
            item.append(record[1])
            last_item.append(record[2])
            hist.append(record[3])
            next_hist.append(record[4])
            click.append(record[5])
        return self.i, (user, item, last_item, hist, next_hist, click)

def evaluate(sess, model, test_set, batch_size):
    auc, embedding_table = [], []
    for _, uij in DataInput(test_set, batch_size):
        label, distance, user, item, embedding = model.test(sess, uij)
        for index in range(len(distance)):
            embedding_table.append(embedding[index])
            if label[index]==distance[index]:
                auc.append(1)
            else:
                auc.append(0)
    return np.mean(auc), embedding_table
    
def recommendation(embedding_table, item_emb_table):
    newtrain_set, new_gini_set = [], []
    tree = scipy.spatial.KDTree(item_emb_table)
    for user, user_embedding in enumerate(embedding_table):
        d, i  = tree.query(user_embedding,k=11)
        new_gini_set = new_gini_set + list(i)
        newtrain_set.append((user, i[9], i[10], i[0:10], i[1:11], 1.0))
    return newtrain_set, new_gini_set

def compute_gini(gini_item_set):
    item_dict = dict(Counter(gini_item_set))
    items = list(item_dict.values())
    items.sort()
    cum_wealths = np.cumsum(sorted(np.append(items, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)
