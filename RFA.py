
# From sangiy92's repo on github
from __future__ import print_function
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def save_stuff(x, y, filename):
    # bigarray = np.column_stack((x,y))
    np.savetxt(filename+'_x', x)
    np.savetxt(filename+'_y', y)
    print("data written to filebase: ",filename)

def scan_lr():
  values = np.arange(0.005, 0.03, 0.005)
  run_scan('lr', values)

def lessfeatures_scanlr():
  values = np.array([128, 256, 512])
  run_scan('num_hidden', values)

def batchnorm_run():
  values = np.arange(0.005, 0.03, 0.005)
  run_scan('lr', values)

def test_batchnorm():
  values = np.array([128])
  run_scan('num_hidden', values)
  # kwargs = {num_hidden:}


def run_scan(param_name=None, values=None, **kwargs):
  # kwargs = {param_name:None}
  kwargs = kwargs or {}
  kwargs[param_name] = None
  for v in values:
    kwargs[param_name] = v
    kwargs["resultdir"] = "results-regular-%s-%f" % (param_name, v)
    kwargs["num_steps"] = 10001
    run_computation(**kwargs)

def run_computation(resultdir="results", lr=0.001, num_steps=10001, back_uni_range=0.5, 
  num_layer=3, num_hidden=1024):
  print("config: ", "lr: ", lr, "num_hidden: ", num_hidden)
  print("starting computation, resultdir: ", resultdir)
  os.mkdir(resultdir)
  # num_steps = 1001


  image_size = 28
  batch_size = 128
  valid_size = test_size = 10000
  num_data_input = image_size*image_size
  num_hidden = 128
  # num_hidden = 1024
  # num_hidden = 100
  num_labels = 10
  act_f = "relu"
  init_f = "uniform"
  back_init_f = "uniform"
  weight_uni_range = 0.05
  # back_uni_range = 0.5
  # lr = 0.001
  # num_layer = 3 #should be >= 3
  # num_steps = 20001
  pickle_file = 'notMNIST.pickle'

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


  # In[11]:

  image_size = 28
  num_labels = 10

  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

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


  # In[12]:

  def drelu(x):
      zero = tf.zeros(x.get_shape())
      one = tf.ones(x.get_shape())
      return(tf.where(tf.greater(x, zero), one, zero))

  def dtanh(x):
      return(1-tf.multiply(tf.nn.tanh(x),tf.nn.tanh(x)))

  def act_ftn(name):
      if(name == "tanh"):
          return(tf.nn.tanh)
      elif(name == "relu"):
          return(tf.nn.relu)
      else:
          print("not tanh or relu")
          
  def dact_ftn(name):
      if(name == "tanh"):
          return(dtanh)
      elif(name == "relu"):
          return(drelu)
      else:
          print("not tanh or relu")

  def init_ftn(name, num_input, num_output, runiform_range):
      if(name == "normal"):
          return(tf.truncated_normal([num_input, num_output]))
      elif(name == "uniform"):
          return(tf.random_uniform([num_input, num_output], minval = -runiform_range, maxval = runiform_range ))
      else:
          print("not normal or uniform")


  # In[13]:

  class Weights:
      def __init__(self, batch_size, num_input, num_output, 
                   act_f, init_f, notfinal = True, back_init_f = "uniform", 
                   weight_uni_range = 0.05, back_uni_range = 0.5):
          self.weights = tf.Variable(init_ftn(init_f, num_input, num_output, weight_uni_range))
          self.biases = tf.Variable(tf.zeros([num_output]))
          backward_t = tf.Variable(init_ftn(back_init_f, num_output, num_input, back_uni_range))
          self.backprop = tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [batch_size, num_output, num_input])
          
          self.batch_size = batch_size
          self.num_input = num_input
          self.num_output = num_output

          self.activation = act_ftn(act_f)
          self.dactivation = dact_ftn(act_f)
          self.notfinal = notfinal

          self.inputs = None
          self.before_activation = None
      
      def __call__(self, x, batch_size):
          if (batch_size == self.batch_size):
              self.inputs = tf.reshape(x, [batch_size, self.num_input, 1])
              self.before_activation = tf.matmul(x, self.weights) + self.biases
              if (self.notfinal):
                  return(self.activation(self.before_activation))
              else:
                  return(self.before_activation)
          else:
              before_activation = tf.matmul(x, self.weights) + self.biases
              if (self.notfinal):
                  return(self.activation(before_activation))
              else:
                  return(before_activation)
      
      def RFA_optimize(self, delta, upper_backprop = None, lr = 0.01):
          if (self.notfinal):
              dError_dhidden = tf.matmul(delta, 
                                           tf.matmul(upper_backprop, tf.matrix_diag(self.dactivation(self.before_activation))))
          else:
              dError_dhidden = delta
          delta_weights = tf.reduce_mean(tf.matmul(self.inputs, dError_dhidden), 0)
          delta_biases = tf.reduce_mean(dError_dhidden, 0)
          change_weights = tf.assign_sub(self.weights, lr*delta_weights)
          change_biases = tf.assign_sub(self.biases, lr*tf.reshape(delta_biases,(self.num_output,)))
          return change_weights, change_biases, dError_dhidden, self.backprop


  # In[14]:

  # hyper parameter setting



  # In[15]:

  graph = tf.Graph()

  with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      
      # model building
      Weight_list = {}

      name = "W0"
      Weight_list[name] = Weights(batch_size, num_data_input, num_hidden, act_f, init_f, True, back_init_f, weight_uni_range, back_uni_range)

      for i in range(num_layer-3):
          name = "W" + str(i+1)
          Weight_list[name] = Weights(batch_size, num_hidden, num_hidden, act_f, init_f, True, back_init_f, weight_uni_range, back_uni_range)

      

      name = "W" + str(num_layer-2)
      Weight_list[name] = Weights(batch_size, num_hidden, num_labels, act_f, init_f, False, back_init_f, weight_uni_range, back_uni_range)

      y_train = None
      x_train = tf_train_dataset
      for i in range(num_layer-1):
          name = "W"+str(i)
          if (i != num_layer - 2):
              x_train = Weight_list[name](x_train, batch_size)
              bn = tf.contrib.layers.batch_norm(x_train, batch_size, center=True, scale=True, is_training=True)
              x_train = bn
          else:
              y_train = Weight_list[name](x_train, batch_size)
      logits = y_train
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)
      loss = tf.reduce_mean(cross_entropy)
      
      dError_dy = tf.reshape(tf.gradients(cross_entropy, logits)[0], [batch_size, 1, num_labels])
      
      # optimize
      train_list = []
      name = "W"+str(num_layer-2)
      upper_backprop = None
      change_weights, change_biases, dError_dhidden, upper_backprop = Weight_list[name].RFA_optimize(dError_dy, upper_backprop, lr)
      train_list += [change_weights, change_biases]
      for i in reversed(range(num_layer-1)):
          name = "W"+str(i)
          change_weights, change_biases, dError_dhidden, upper_backprop = Weight_list[name].RFA_optimize(dError_dhidden, upper_backprop, lr)
          train_list += [change_weights, change_biases]

      y_valid = None
      x_valid = tf_valid_dataset
      for i in range(num_layer-1):
          name = "W"+str(i)
          if (i != num_layer - 2):
              x_valid = Weight_list[name](x_valid, valid_size)
          else:
              y_valid = Weight_list[name](x_valid, valid_size)
      logits_valid = y_valid
      
      y_test = None
      x_test = tf_test_dataset
      for i in range(num_layer-1):
          name = "W"+str(i)
          if (i != num_layer - 2):
              x_test = Weight_list[name](x_test, test_size)
              bn = tf.contrib.layers.batch_norm(x_test, test_size, center=True, scale=True, is_training=False)
              x_test = bn
          else:
              y_test = Weight_list[name](x_test, test_size)
      logits_test = y_test
      
      # Predictions for the training, validation, and test data.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_prediction = tf.nn.softmax(logits)
      # train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(logits_valid)
      test_prediction = tf.nn.softmax(logits_test)

      # with tf.name_scope("test_accuracy"):
      #   tf.summary.scalar('test_acc', test_prediction)

      # merged = tf.summary.merge_all()
      


  # In[ ]:

  with tf.Session(graph=graph) as session:
    # fwriter = tf.summary.FileWriter(resultdir, session.graph)
    tf.global_variables_initializer().run()
    outfilename = os.path.join(resultdir, "output.txt")
    outfile = open(outfilename, "w")
    outfile.write("Initialized")
    x = []
    y = []
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
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
      session.run(train_list, feed_dict = feed_dict)
      if (step % 500 == 0):
        print("Step: ", step)
        outfile.write("Minibatch loss at step %d: %f" % (step, l))
        outfile.write("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        outfile.write("Validation accuracy: %.1f%%" % accuracy(
          valid_prediction.eval(), valid_labels))
        test_acc = accuracy(test_prediction.eval(), test_labels)
        x.append(step)
        y.append(test_acc)
        # summary, _ = session.run(merged, feed_dict=feed_dict)
        # fwriter.add_summary(summary, step)

    outfile.write("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    save_stuff(np.array(x), np.array(y), outfilename)



if __name__ == "__main__":
  fn = sys.argv[1]
  locals()[fn]()
