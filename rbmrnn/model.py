# -*- coding:utf-8 -*-

import math
import random
import numpy
import tensorflow as tf


def normal_variables(shape, stddev=0.01):
	return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))


def zero_variables(shape):
	return tf.Variable(tf.zeros(shape=shape))


def rbm_model(v, W, b, c, k):
	# E(v, h) = -bv - ch - hWv
	# v [batch_size, n_visible]
	# W [n_visible, n_hidden]
	# b [batch_size, n_visible]
	# c [batch_size, n_hidden]

	def gibbs_step(v_t, b_t, c_t):
		h_mean = tf.sigmoid(tf.matmul(v_t, W) + c_t)
		noise = numpy.random.uniform(0, 1, size=h_mean.shape)
		h_sample = tf.cast(tf.greater(h_mean, noise), 'float')
		v_mean = tf.sigmoid(tf.matmul(h_sample, tf.transpose(W, [1, 0])) + b_t)
		noise = numpy.random.uniform(0, 1, size=v_mean.shape)
		v_sample = tf.cast(tf.greater(v_mean, noise), 'float')
		v_sample = tf.stop_gradient(v_sample)
		return v_sample

	v_splitted = tf.split(v, int(v.shape[0]), 0)   # can be a single frame
	v_samples = None
	for step, v_t in enumerate(v_splitted):
		for _ in xrange(k):
			v_sample = gibbs_step(v_t, b[step], c[step])
			v_t = v_sample
		v_samples = tf.concat([v_samples, v_sample], 0) if v_samples is not None else v_sample

	def free_energy(v):
		temp = tf.clip_by_value(tf.matmul(v, W) + c, float('-inf'), 10)
		return tf.reduce_mean(v * b, 1) - tf.reduce_mean(tf.log(1 + tf.exp(temp)), 1)

	# [warning] stop gradients flow into gibbs_step
	v_samples = tf.stop_gradient(v_samples)
	cost = tf.reduce_mean(free_energy(v) - free_energy(v_samples))

	# debug = b, c, v * b, tf.matmul(v, W), tf.exp(tf.matmul(v, W) + c)
	debug = None

	return v_samples, cost, debug


def rnnrbm_model(v, n_visible, n_hidden, n_hidden_rnn, lr, generation_steps):

	W = normal_variables([n_visible, n_hidden], 0.01)
	Wvu = normal_variables([n_visible, n_hidden_rnn], 0.0001)
	Wuv = normal_variables([n_hidden_rnn, n_visible], 0.0001)
	Wuh = normal_variables([n_hidden_rnn, n_hidden], 0.0001)
	Wuu = normal_variables([n_hidden_rnn, n_hidden_rnn], 0.0001)
	bv = zero_variables([n_visible])
	bh = zero_variables([n_hidden])
	bu = zero_variables([n_hidden_rnn])
	params = [W, Wvu, Wuv, Wuh, Wuu, bv, bh, bu]

	def recurrence(v_t, u_old, generate=False):
		b_t = tf.matmul(u_old, Wuv) + bv
		c_t = tf.matmul(u_old, Wuh) + bh
		if generate:
			v_t, _, _ = rbm_model(zero_variables([1, n_visible]), W, b_t, c_t, k=15)
		u_t = tf.tanh(tf.matmul(v_t, Wvu) + tf.matmul(u_old, Wuu) + bu)
		if generate:
			return v_t, u_t
		else:
			return u_t, b_t, c_t

	# training model
	v_splitted = tf.split(v, int(v.shape[0]), 0)
	u_t = zero_variables([1, n_hidden_rnn])
	b = None
	c = None
	for step, v_t in enumerate(v_splitted):
		u_t, b_t, c_t = recurrence(v_t, u_t, generate=False)
		b = tf.concat([b, b_t], 0) if b is not None else b_t
		c = tf.concat([c, c_t], 0) if c is not None else c_t
	_, cost, debug = rbm_model(v, W, b, c, k=5)
	train = tf.train.AdamOptimizer(lr).minimize(cost, var_list=params)

	# generation model
	vg = None
	vg_t = None
	ug_t = zero_variables([1, n_hidden_rnn])
	for step in xrange(generation_steps):
		vg_t, ug_t = recurrence(vg_t, ug_t, generate=True)
		vg = tf.concat([vg, vg_t], 0) if vg is not None else vg_t
	return cost, train, vg, params, debug

