#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 drop_rate=0.5,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):

        n_input = 3072
        n_output=10
        n_hidden_1 = 256

        self.x_ = tf.placeholder(tf.float32, [None, n_input])
        self.y_ = tf.placeholder(tf.int32, [None])

        self.drop_rate = drop_rate
        self.weights = {
            "h1": tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
            "out": tf.Variable(tf.constant(0.0, shape=[n_hidden_1, n_output])),
        }
        self.biases = {
            "h1": tf.Variable(tf.random_normal([n_hidden_1])),
            "out": tf.Variable(tf.random_normal([n_output])),
        }

        self.loss, self.pred, self.acc = self.forward(True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(False, reuse=True)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
                                    
    def forward(self, is_train, reuse=None):
        with tf.variable_scope("model", reuse=reuse):
            '''
            implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            the 10-class prediction output is named as "logits"
            '''
            # Linear Layer
            hidden_1 = tf.add(tf.matmul(self.x_, self.weights["h1"]), self.biases["h1"])

            # BN Layer: use batch_normalization_layer function
            BN_layer = batch_normalization_layer(hidden_1, is_train=True)

            # Relu Layer
            Relu_layer = tf.nn.relu(BN_layer)

            # Dropout Layer: use dropout_layer function
            DO_layer = dropout_layer(Relu_layer, self.drop_rate, is_train=is_train)

            # Linear Layer
            logits = tf.add(tf.matmul(DO_layer, self.weights["out"]), self.biases["out"])  # deleted this line after you implement above layers

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, is_train=True):
    '''
    implement the batch normalization function and applied it on fully-connected layers
    NOTE:  If isTrain is True, you should use mu and sigma calculated based on mini-batch
          If isTrain is False, you must use mu and sigma estimated from training data
    '''
    outgoing = tf.layers.batch_normalization(incoming, training=is_train, momentum=0.9)
    return outgoing
    
def dropout_layer(incoming, drop_rate, is_train=True, alternative=False):
    '''
    implement the dropout function and applied it on fully-connected layers
    NOTE: When drop_rate=0, it means drop no values
          If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
          If isTrain is False, remain all values not changed
    '''
    if is_train:
        if alternative:
            outgoing = tf.scalar_mul(1/(1-drop_rate), incoming)
        else:
            outgoing = tf.nn.dropout(incoming, rate=drop_rate)
    else:
        outgoing = incoming
    return outgoing

