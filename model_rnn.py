import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

from tensorflow.python.ops import seq2seq

class rnn_model(object):
    """
    A RNN for predicting output based on long term dependancies.
    """
    def __init__(self, is_training, data_size, num_layers, stddev_init, batch_size, state_size, num_steps, num_classes, cell_type = 'LSTM', tf_ = False):
        
        if cell_type == 'BASIC':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif cell_type == 'GRU':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif cell_type == 'LSTM':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(cell_type))

        if cell_type == 'blabla':
            cell = cell_fn(state_size, state_is_tuple = True)
        else: 
            cell = cell_fn(state_size)

        if is_training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)

        if cell_type == 'blabla': 
            self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple = True)
        else: 
            self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        self.x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
        self.y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        W_softmax = tf.Variable(tf.truncated_normal([state_size, num_classes], stddev = stddev_init[1]))
        b_softmax = tf.Variable(tf.zeros([num_classes]))

        embeddings = tf.Variable(tf.random_uniform([num_classes, state_size], -1.0, 1.0), name = "W")

        rnn_inputs = [tf.squeeze(i) for i in tf.split(1, num_steps, tf.nn.embedding_lookup(embeddings, self.x))]
        rnn_outputs, last_state = tf.nn.rnn(cell, rnn_inputs, self.initial_state)

        self.final_state = last_state
        self.logits = [tf.matmul(rnn_output, W_softmax) + b_softmax for rnn_output in rnn_outputs]
        self.probs = [tf.nn.softmax(logit) for logit in self.logits]

        # Turn our y placeholder into a list labels
        y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, self.y)]
        
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit,label) for logit, label in zip(self.logits, y_as_list)]
        self.loss = tf.reduce_mean(self.losses)

        self.predictions_label = [tf.argmax(probs, 1, name="predictions") for probs in self.probs]
        # Accuracy
        self.correct_predictions = [tf.equal(self.predictions_label, tf.argmax(self.y, 1))]
        self.global_accuracy = [tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy") for correct_prediction in self.correct_predictions]
        self.accuracy = tf.reduce_mean(self.global_accuracy)
