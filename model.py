# -*- coding:utf-8 -*-

import random
import numpy
import tensorflow as tf


def lstm(x, state, output, n_steps, n_input, n_hidden, n_output, learning_rate):
	# Parameters:
	# Input gate: input, previous output, and bias
	ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	ib = tf.Variable(tf.zeros([1, n_hidden]))
	# Forget gate: input, previous output, and bias
	fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	fb = tf.Variable(tf.zeros([1, n_hidden]))
	# Memory cell: input, state, and bias
	cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	cb = tf.Variable(tf.zeros([1, n_hidden]))
	# Output gate: input, previous output, and bias
	ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
	om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	ob = tf.Variable(tf.zeros([1, n_hidden]))

	# Generation layer
	gw = tf.Variable(tf.truncated_normal([n_hidden, n_output]))
	gb = tf.Variable(tf.zeros([1, n_output]))

	# Definition of the cell computation
	def lstm_cell(i_cur, o_pre, state):
		input_gate = tf.sigmoid(tf.matmul(i_cur, ix) + tf.matmul(o_pre, im) + ib)
		forget_gate = tf.sigmoid(tf.matmul(i_cur, fx) + tf.matmul(o_pre, fm) + fb)
		update = tf.tanh(tf.matmul(i_cur, cx) + tf.matmul(o_pre, cm) + cb)
		state = forget_gate * state + input_gate * update
		output_gate = tf.sigmoid(tf.matmul(i_cur, ox) + tf.matmul(o_pre, om) + ob)
		return output_gate * tf.tanh(state), state

	def generate_output(lstm_output):
		return tf.matmul(lstm_output, gw) + gb

	# Unrolled LSTM loop
	outputs_training = list()
	outputs_generation = list()

	# x shape: (batch_size, n_steps, n_input)
	# desired shape: list of n_steps with element shape (batch_size, n_input)
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_steps, 0)
	cost = 0
	for index in xrange(n_steps):
		output, state = lstm_cell(x[index], output, state)
		outputs_training.append(generate_output(output))
		if index > 0:
			cost += tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(labels=x[index], logits=outputs_training[-2])
			)
	train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# generation
	i_g = tf.Variable(tf.zeros([1, n_input]))
	# random input
	r_position = random.randint(0, n_input - 1)
	i_g[0, r_position].assign(1)

	state_g = tf.Variable(tf.zeros([1, n_hidden]))
	output_g = tf.Variable(tf.zeros([1, n_hidden]))
	for _ in xrange(100):
		output_g, state_g = lstm_cell(i_g, output_g, state_g)
		output_sigmoid = tf.nn.sigmoid(generate_output(output_g))
		outputs_generation.append(tf.cast(tf.greater(output_sigmoid, 0.5), 'float'))
		i_g = output_sigmoid

	return cost, train, outputs_generation

