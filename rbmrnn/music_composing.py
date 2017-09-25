# -*- coding:utf-8 -*-

import glob
import model
import numpy
import tensorflow as tf
from midi.utils import midiread, midiwrite


class MIDIFileMC(object):

	def __init__(
			self, n_hidden=150, n_hidden_rnn=100, r=(21, 109), dt=0.3,
			batch_size=100, lr=0.001, generation_steps=200):
		self.lr = lr
		self.r = r
		self.dt = dt
		self.v = tf.placeholder(tf.float32, [batch_size, r[1] - r[0]])
		self.cost, self.train_op, self.vg, self.params, self.debug = model.rnnrbm_model(
			self.v, r[1] - r[0], n_hidden, n_hidden_rnn, lr, generation_steps)
		self.batch_size = batch_size
		self.train_files = glob.glob(r'Nottingham/train/*.mid')
		self.generation_path = 'generated/'

	def train(self, session, num_epochs=200):
		dataset = [midiread(f, self.r, self.dt).piano_roll.astype(numpy.float32) for f in self.train_files][0: 5]
		num_samples = len(dataset)
		print 'total samples: %d' % num_samples
		for epoch in xrange(num_epochs):
			numpy.random.shuffle(dataset)
			for s, sequence in enumerate(dataset):
				total_cost = list()
				for i in range(0, len(sequence) - self.batch_size + 1, self.batch_size):
					v_batch = sequence[i: i + self.batch_size]
					train_result = session.run([self.cost, self.train_op], feed_dict={self.v: v_batch})
					total_cost.append(train_result[0])

				if s % 100 == 0:
					if len(total_cost) > 0:
						mean_cost = sum(total_cost) / float(len(total_cost))
						print 'epoch %d, %dth sample length %d, cost: %f' % (epoch, s, len(sequence), mean_cost)
					# for testing
					# params = session.run(self.params)
					# print [numpy.mean(p) for p in params]
					# print [numpy.amax(numpy.abs(p)) for p in params]
					# print [numpy.amax(numpy.abs(train_result[2]))]

	def generate(self, session, filename='001.mid'):
		sequence = session.run(self.vg)
		midiwrite(self.generation_path + filename, sequence, self.r, self.dt)


if __name__ == '__main__':
	mc = MIDIFileMC(batch_size=50, lr=0.001, generation_steps=100)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		mc.train(sess, num_epochs=1000)
		mc.generate(sess)

