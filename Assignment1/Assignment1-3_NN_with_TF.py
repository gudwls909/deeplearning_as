
# coding: utf-8

# # M2177.003100 Deep Learning <br> Assignment #1 Part 3: Playing with Neural Networks by TensorFlow

# Copyright (C) Data Science & AI Laboratory, Seoul National University. This material is for educational uses only. Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 

# Previously in `Assignment2-1_Data_Curation.ipynb`, we created a pickle with formatted datasets for training, development and testing on the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
# 
# The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow.
# 
# **Note**: certain details are missing or ambiguous on purpose, in order to test your knowledge on the related materials. However, if you really feel that something essential is missing and cannot proceed to the next step, then contact the teaching staff with clear description of your problem.
# 
# ### Submitting your work:
# <font color=red>**DO NOT clear the final outputs**</font> so that TAs can grade both your code and results.  
# Once you have done **part 1 - 3**, run the *CollectSubmission.sh* script with your **Student number** as input argument. <br>
# This will produce a compressed file called *[Your student number].tar.gz*. Please submit this file on ETL. &nbsp;&nbsp; (Usage: ./*CollectSubmission.sh* &nbsp; 20\*\*-\*\*\*\*\*)

# ## Load datasets
# 
# First reload the data we generated in `Assignment2-1_Data_Curation.ipynb`.

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os

#configuration for gpu usage
conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.4
conf.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES']='0'


# In[2]:


pickle_file = 'data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[3]:


image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# ## TensorFlow tutorial: Fully Connected Network
# 
# We're first going to train a **fully connected network** with *1 hidden layer* with *1024 units* using stochastic gradient descent (SGD).
# 
# TensorFlow works like this:
# * First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. This description is all contained within the block below:
# 
#       with graph.as_default():
#           ...
# 
# * Then you can run the operations on this graph as many times as you want by calling `session.run()`, providing it outputs to fetch from the graph that get returned. This runtime operation is all contained in the block below:
# 
#       with tf.Session(graph=graph) as session:
#           ...
# 
# Let's load all the data into TensorFlow and build the computation graph corresponding to our training:

# In[4]:


batch_size = 128
nn_hidden = 1024

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_dataset = tf.placeholder(tf.float32,
                                      shape=(None, image_size * image_size))
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    
    # Variables. 
    w1 = tf.Variable(tf.truncated_normal([image_size * image_size, nn_hidden]))
    b1 = tf.Variable(tf.zeros([nn_hidden]))
    w2 = tf.Variable(tf.truncated_normal([nn_hidden, num_labels]))
    b2 = tf.Variable(tf.zeros([num_labels]))
    
    # Training computation.
    hidden = tf.tanh(tf.matmul(tf_dataset, w1) + b1)
    logits = tf.matmul(hidden, w2) + b2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)


# Let's run this computation and iterate:

# In[5]:


num_steps = 10000

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.equal(np.argmax(predictions, 1), np.argmax(labels, 1)))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict_train={tf_dataset: batch_data, tf_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict_train)
        if (step % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            valid_prediction = session.run(logits, feed_dict={tf_dataset: valid_dataset})
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction, valid_labels))
                  
    test_prediction = session.run(prediction, feed_dict={tf_dataset: test_dataset})
    print("Test accuracy: %.1f%%" % accuracy(test_prediction, test_labels))
    saver = tf.train.Saver()
    saver.save(session, "./model_checkpoints/my_model_final")


# So far, you have built the model in a naive way. However, TensorFlow provides a module named tf.layers for your convenience. 
# 
# From now on, build the same model as above using layers module.

# In[6]:


graph_l=tf.Graph()
with graph_l.as_default():
    tf_dataset_l=tf.placeholder(tf.float32, shape=(None, image_size * image_size))
    tf_labels_l=tf.placeholder(tf.float32, shape=(None, num_labels))
    
    dense = tf.layers.dense(tf_dataset_l, nn_hidden, activation=tf.tanh)
    logits_l = tf.layers.dense(dense, num_labels, activation=tf.nn.softmax)
    
    #Loss
    loss_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_labels_l, logits=logits_l))
    
    #Optimizer
    optimizer_l = tf.train.GradientDescentOptimizer(0.5).minimize(loss_l)
    
    #Predictions for the training
    prediction_l = tf.nn.softmax(logits_l)


# In[7]:


with tf.Session(graph=graph_l, config=conf) as session_l:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :].astype(float)
        feed_dict_l = {tf_dataset_l: batch_data, tf_labels_l: batch_labels}
        _, l_l, predictions_l = session_l.run([optimizer_l, loss_l, prediction_l], feed_dict=feed_dict_l)
        if(step % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l_l))
            feed_dict_val_l = {tf_dataset_l: valid_dataset}
            valid_prediction_l = session_l.run(prediction_l, feed_dict={tf_dataset_l: valid_dataset, tf_labels_l: valid_labels})
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction_l, valid_labels))

    feed_dict_test_l = {tf_dataset_l: test_dataset}
    test_prediction_l = session_l.run(prediction_l, feed_dict=feed_dict_test_l)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction_l, test_labels))
    saver = tf.train.Saver()
    saver.save(session_l, "./model_checkpoints/my_model_final_using_layers")


# ---
# Problem 1
# -------
# 
# **Describe below** why there is a difference in an accuracy between the graph using layer module and the graph which is built in a naive way.
# 
# 
# 
# 
# 
# ---

# naive하게 만든 모델의 경우에는 truncated_normal로 initialization을 했고, layer module에서는 kernel_initializer가 따로 입력되지 않은 경우에는 glorot_uniform_initializer로 initialization을 한다. 즉, parameter의 initialization이 다르기 때문에 결과가 다르게 나왔다.

# ---
# Problem 2
# -------
# 
# Try to get the best performance you can using a multi-layer model! (It doesn't matter whether you implement it in a naive way or using layer module. HOWEVER, you CANNOT use other type of layers such as conv.) 
# 
# The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.kr/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595). You may use techniques below.
# 
# 1. Experiment with different hyperparameters: num_steps, learning rate, etc.
# 2. We used a fixed learning rate $\epsilon$ for gradient descent. Implement an annealing schedule for the gradient descent learning rate ([more info](http://cs231n.github.io/neural-networks-3/#anneal)). *Hint*. Try using `tf.train.exponential_decay`.    
# 3. We used a $\tanh$ activation function for our hidden layer. Experiment with other activation functions included in TensorFlow.
# 4. Extend the network to multiple hidden layers. Experiment with the layer sizes. Adding another hidden layer means you will need to adjust the code. 
# 5. Introduce and tune regularization method (e.g. L2 regularization) for your model. Remeber that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should imporve your validation / test accuracy.
# 6. Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
# 
# **Evaluation:** You will get full credit if your best test accuracy exceeds 93%. Save your best perfoming model as my_model_final using saver. (Refer to the cell above) 
# 
# ---

# In[ ]:


# regularization (l2_loss)
# decaying learning rate (tf.train.exponential_decay)
# activation function (ReLU)
# tuning hyperparameters (num_steps, learning rate..)
# deeper (hidden_layer1 100, hidden_layer2 100)
# dropout (keep_prob 0.9)


# In[14]:


batch_size = 128
nn_hidden_1, nn_hidden_2 = 100, 100
num_steps = 20000
epsilon = 0.8

graph_l=tf.Graph()
with graph_l.as_default():
    tf_dataset_l=tf.placeholder(tf.float32, shape=(None, image_size * image_size))
    tf_labels_l=tf.placeholder(tf.float32, shape=(None, num_labels))
    
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
    hidden1 = tf.layers.dense(tf_dataset_l, nn_hidden_1, activation=tf.nn.relu, kernel_regularizer=regularizer)
    hidden1_dropout = tf.nn.dropout(hidden1, 0.9)
    hidden2= tf.layers.dense(hidden1_dropout, nn_hidden_2, activation=tf.nn.relu, kernel_regularizer=regularizer)
    logits_l = tf.layers.dense(hidden2, num_labels, activation=tf.nn.softmax, kernel_regularizer=regularizer)
    #dense = tf.layers.dense(tf_dataset_l, nn_hidden, activation=tf.nn.relu)
    #logits_l = tf.layers.dense(dense, num_labels, activation=tf.nn.softmax)
    
    loss_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_labels_l, logits=logits_l))
    l2_loss = tf.losses.get_regularization_loss()
    loss_l += l2_loss
    
    learning_rate = tf.train.exponential_decay(epsilon, num_steps, 2000, 0.95, staircase=True)
    learning_rate = epsilon
    optimizer_l = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_l)
    
    #Predictions for the training
    prediction_l = tf.nn.softmax(logits_l)


# In[15]:


with tf.Session(graph=graph_l, config=conf) as session_l:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :].astype(float)
        feed_dict_l = {tf_dataset_l: batch_data, tf_labels_l: batch_labels}
        _, l_l, predictions_l = session_l.run([optimizer_l, loss_l, prediction_l], feed_dict=feed_dict_l)
        if(step % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l_l))
            train_prediction_l = session_l.run(prediction_l, feed_dict={tf_dataset_l: train_dataset, tf_labels_l: train_labels})
            print('Train accuracy: %.1f%%' % accuracy(train_prediction_l, train_labels))
            
            valid_prediction_l = session_l.run(prediction_l, feed_dict={tf_dataset_l: valid_dataset, tf_labels_l: valid_labels})
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction_l, valid_labels))

    feed_dict_test_l = {tf_dataset_l: test_dataset}
    test_prediction_l = session_l.run(prediction_l, feed_dict=feed_dict_test_l)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction_l, test_labels))
    saver = tf.train.Saver()
    saver.save(session_l, "./model_checkpoints/my_model_final")

