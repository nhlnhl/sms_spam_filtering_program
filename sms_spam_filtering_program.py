
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from collections import Counter
import random

tf.set_random_seed(777)

data = np.genfromtxt('SMSSpamCollection', names=('type', 'text'), dtype=None, delimiter='\t', usecols=(0,1))
data = np.array(data)

print(data)
print(data.shape)

np.random.shuffle(data)

print(data)
print(data.shape)


# In[2]:


x_data = data['text']
y_data = data['type']

x_data = np.reshape(x_data, (-1, 1))
y_data = np.reshape(y_data, (-1, 1))

print(x_data)
print(x_data.shape)
print(y_data)
print(y_data.shape)

train_size = int(len(y_data) * 0.8)
test_size = len(y_data) - train_size

x_data_test = np.array(x_data[train_size:len(x_data)])
x_data = np.array(x_data[0:train_size])
y_data_test = np.array(y_data[train_size:len(y_data)])
y_data = np.array(y_data[0:train_size])

vocab = Counter()

for text in x_data:
    for word in np.array2string(text).split(' '):
        vocab[word.lower()] += 1
        
total_words = len(vocab)
print("Total words:", total_words)

def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i
        
    return word2index

word2index = get_word_2_index(vocab)

x_data_input = []
for text in x_data:
    layer = np.zeros(total_words, dtype=float)
    for word in np.array2string(text).split(' '):
        layer[word2index[word.lower()]] += 1
            
    x_data_input.append(layer)
    
y_data_output = []
for category in y_data:
    y = np.zeros((2), dtype=float)
    if category == [b'ham']:
        y[0] = 1.
    else:
        y[1] = 1.
    
    y_data_output.append(y)
    
x_data_input = np.array(x_data_input)
y_data_output = np.array(y_data_output)

print(x_data_input)
print(y_data_output)


# In[3]:


n_input = total_words
n_classes = 2
n_hidden_1 = 100
n_hidden_2 = 100

x_input = tf.placeholder(tf.float32, [None, n_input], name="x_input")
y_output = tf.placeholder(tf.float32, [None, n_classes], name="y_output")

learning_rate = 0.01
training_epochs = 10


# In[4]:


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['w1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['w2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)
    
    # Output layer 
    out_layer_multiplication = tf.matmul(layer_2, weights['w3'])
    out_layer_addition = out_layer_multiplication + biases['b3']
    
    return out_layer_addition


# In[5]:


weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_classes]))
}


# In[6]:


prediction = multilayer_perceptron(x_input, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()


# In[ ]:


with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        c, _ = sess.run([cost, optimizer], feed_dict={x_input: x_data_input, y_output: y_data_output})
        print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(c))
    
    x_data_input_test = []
    for text in x_data_test:
        layer = np.zeros(total_words, dtype=float)
        for word in np.array2string(text).split(' '):
            if word.lower() in vocab:
                layer[word2index[word.lower()]] += 1
            
    x_data_input_test.append(layer)
    
    y_data_output_test = []
    for category in y_data_test:
        y = np.zeros((2), dtype=float)
        if category == [b'ham']:
            y[0] = 1.
        else:
            y[1] = 1.
    
    y_data_output_test.append(y)
    
    x_data_input_test = np.array(x_data_input_test)
    y_data_output_test = np.array(y_data_output_test)
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_data_output_test, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x_input: x_data_input_test, y_output: y_data_output_test}))
    
    while True:
        str = input("Type your text message: ")
    
        x_test = np.zeros(total_words, dtype=float)
        for word in str.split(' '):
            if word.lower() in vocab:
                x_test[word2index[word.lower()]] += 1
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (1, -1))
    
        y_test = sess.run(tf.argmax(prediction, 1), feed_dict={x_input: x_test})
    
        if(y_test == 0):
            print("'", str, "' is a ham message.")
        else:
            print("'", str, "' is a spam message.")

