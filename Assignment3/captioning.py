# This library is used for Assignment3_Part2_ImageCaptioning

# Write your own image captioning code
# You can modify the class structure
# and add additional function needed for image captioning

import tensorflow as tf
import numpy as np


class Captioning():
	
	def __init__(self, word_to_idx, idx_to_word, input_dim=512, hidden_dim=128, n_words=1004, maxlen=17, emb_dim=128):

		self.word_to_idx = word_to_idx
		self.idx_to_word = idx_to_word
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.n_words = n_words
		self.maxlen = maxlen
		self.emb_dim = emb_dim

		"""
		self.params = dict()

		self.params['W_init'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
		self.params['b_init'] = np.zeros(hidden_dim)

		self.params['W_embed'] = np.random.randn(n_words, emb_dim) / np.sqrt(n_words)

		self.params['Wx'] = np.random.randn(emb_dim, 4*hidden_dim) / np.sqrt(emb_dim)
		self.params['Wh'] = np.random.randn(hidden_dim, 4*hidden_dim) / np.sqrt(hidden_dim)
		self.params['bx'] = np.random.randn(4*hidden_dim)

		self.params['Wo'] = np.random.randn(hidden_dim, n_words) / np.sqrt(hidden_dim)
		self.params['b0'] = np.random.randn(n_words)

		for key, value in self.params.items():
			self.params[key] = value.astype(np.float32)
		"""

	def build_model(self):
		graph = tf.Graph()
		with graph.as_default():
			self.inputs = tf.placeholder(tf.int32, [None, self.maxlen - 1])
			self.labels = tf.placeholder(tf.int32, [None, self.maxlen - 1])
			embeddings = tf.get_variable('embedding_matrix', [self.n_words, self.emb_dim])
			self.lstm_inputs = tf.nn.embedding_lookup(embeddings, self.inputs)

			self.features = tf.placeholder(tf.float32, [None, self.input_dim])
			self.W_init = tf.get_variable('W_init', [self.input_dim, self.hidden_dim])
			self.b_init = tf.get_variable('b_init', [self.hidden_dim])
			h_0 = tf.matmul(self.features, self.W_init) + self.b_init
			#h_0 = tf.matmul(self.features, self.params['W_init']) + self.params['b_init']

			lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_dim)
			self.outputs, final_state = tf.nn.dynamic_rnn(lstm, self.lstm_inputs, initial_state=h_0) # h_0 에 CNN에서나온 features를 LSTM의 h_0로 입력하는데 오류
			#self.outputs, final_state = tf.nn.dynamic_rnn(lstm, self.lstm_inputs, dtype=tf.float32)

			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.labels)
			self.loss = tf.reduce_mean(cross_entropy)
			self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

		return graph

	def predict(self):
		captions = None


