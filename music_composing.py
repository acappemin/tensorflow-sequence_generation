# -*- coding:utf-8 -*-

import collections
import glob
import model
import numpy
import tensorflow as tf
from midi.utils import midiread, midiwrite


r = (21, 109)
dt = 0.2
train_files = glob.glob(r'../rbmrnn/Nottingham/train/*.mid')
n_input = r[1] - r[0]
n_hidden = 64
n_output = r[1] - r[0]
n_steps = 50
learning_rate = 0.5
learning_rate_decay = 0.99
batch_size = 32
num_epochs = 2000
check_epochs = 1
generation_path = 'generated/'
filename = '001.mid'

dataset = [midiread(f, r, dt).piano_roll.astype(numpy.float32) for f in train_files]
num_samples = len(dataset)
print 'total samples: %d' % num_samples

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
state = tf.placeholder(tf.float32, [None, n_hidden])
output = tf.placeholder(tf.float32, [None, n_hidden])

learning_rate_variable = tf.Variable(float(learning_rate), trainable=False)
learning_rate_decay_op = learning_rate_variable.assign(learning_rate_variable * learning_rate_decay)
cost, train_op, generator = model.lstm(
	x, state, output, n_steps, n_input, n_hidden, n_output, learning_rate_variable)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	losses = collections.deque(maxlen=3)
	loss_per_check = 0.0
	for epoch in xrange(num_epochs):
		numpy.random.shuffle(dataset)
		# dataset: n_samples * [n_frames, n_input]
		for b in range(0, len(dataset) - batch_size + 1, batch_size):
			data_batch = dataset[b: b + batch_size]
			min_len = min([len(sequence) for sequence in data_batch])
			total_cost = list()
			for i in range(0, min_len - n_steps + 1, n_steps):
				x_batch = numpy.concatenate(
					[sequence[i: i + n_steps].reshape((1, n_steps, n_input)) for sequence in data_batch], 0)
				train_result = sess.run([cost, train_op], feed_dict={
					x: x_batch, state: numpy.zeros([batch_size, n_hidden]),
					output: numpy.zeros([batch_size, n_hidden])})
				total_cost.append(train_result[0])

			if len(total_cost) > 0:
				loss_per_check += sum(total_cost) / float(len(total_cost))

		if epoch % check_epochs == 0:
			print 'epoch %d, epoch_cost: %f' % (epoch, loss_per_check)
			if len(losses) > 2 and loss_per_check > max(losses):
				lr = sess.run(learning_rate_decay_op)
				print 'learning rate decay to %f' % lr
			losses.append(loss_per_check)
			loss_per_check = 0.0

	sequence = sess.run(generator, feed_dict={x: x_batch})
	for index in xrange(len(sequence)):
		sequence[index] = sequence[index].reshape((n_output,))
	print '#############################'
	for t_v in sequence[::5]:
		print numpy.sum(t_v), t_v

midiwrite(generation_path + filename, sequence, r, dt)

