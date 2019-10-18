#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 drop_rate=0.5,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):

        n_input = 32
        n_input_channel = 3

        self.x_ = tf.placeholder(tf.float32, [None, n_input, n_input, n_input_channel])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.drop_rate = drop_rate

        self.loss, self.pred, self.acc = self.forward(True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            '''
            implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
                   the 10-class prediction output is named as "logits"
            '''

            # Conv Layer
            Conv_0 = tf.layers.conv2d(self.x_, filters=10, kernel_size=5)
            # BN Layer: use batch_normalization_layer function
            BN_0 = batch_normalization_layer(Conv_0, is_train=is_train)
            # Relu Layer
            Relu_0 = tf.nn.relu(BN_0)
            # Dropout Layer: use dropout_layer function
            DO_0 = dropout_layer(Relu_0, self.drop_rate, is_train=is_train)
            # MaxPool
            Pool_0 = tf.layers.max_pooling2d(DO_0, pool_size=2, strides=2)
            # Conv Layer
            Conv_1 = tf.layers.conv2d(Pool_0, filters=20, kernel_size=4)
            # BN Layer: use batch_normalization_layer function
            BN_1 = batch_normalization_layer(Conv_1, is_train=is_train)
            # Relu Layer
            Relu_1 = tf.nn.relu(BN_1)
            # Dropout Layer: use dropout_layer function
            DO_1 = dropout_layer(Relu_1, self.drop_rate, is_train=is_train)
            # MaxPool
            Pool_1 = tf.layers.max_pooling2d(DO_1, pool_size=2, strides=2)
            # Linear Layer
            logits = tf.layers.dense(tf.layers.flatten(Pool_1), 10)

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
