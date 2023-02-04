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