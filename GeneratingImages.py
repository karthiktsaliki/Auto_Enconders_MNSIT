import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

#import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

n_pixels=28*28

#input the images (gateway)
X=tf.placeholder(tf.float32,shape=([None,n_pixels]))

def weight_variables(shape,name):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial,name=name)

def bias_variable(shape,name):
	initail=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initail,name=name)

def FC_Layer(X,W,b):
	return tf.matmul(X,W) + b

latent_dim=20
h_dim=500


# ENCODER

#Layer 1
W_enc=weight_variables([n_pixels,h_dim],'W_enc')
b_enc=bias_variable([h_dim],'b_enc')

#Activation Function
h_enc=tf.nn.tanh(FC_Layer(X,W_enc,b_enc ))

#Layer 2
W_mu=weight_variables([h_dim,latent_dim],'W_mu')
b_mu=bias_variable([latent_dim],'b_mu')

#Activation Function
mu=FC_Layer(h_enc,W_mu,b_mu) #mean


#Standard deviation
W_logstd=weight_variables([h_dim,latent_dim],'W_logstd')
b_logstd=bias_variable([latent_dim],'b_logstd')

#Activation Function
logstd=FC_Layer(h_enc,W_logstd,b_logstd) #std


#Randomness
noise=tf.random_normal([1,latent_dim])
z=mu+tf.multiply(noise,tf.exp(.5*logstd))

#DECODER

#Layer 1
W_dec=weight_variables([latent_dim,h_dim],'W_dec')
b_dec=bias_variable([h_dim],'b_dec')

#pass in the z here (and the weights and biases are just defined)
h_dec=tf.nn.tanh(FC_Layer(z,W_dec,b_dec))

#Layer 2
W_reconstruct=weight_variables([h_dim,n_pixels],'W_recontruct')
b_reconstruct=bias_variable([n_pixels],'b_reconstruct')

#bernoulli parameters output
reconstruction=tf.nn.sigmoid(FC_Layer(h_dec,W_reconstruct,b_reconstruct ))

#Loss Function
log_likelihood = tf.reduce_sum(X*tf.log(reconstruction + 1e-9)+(1 - X)*tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

#Training
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver()

import time #lets clock training time..

num_iterations = 10000
recording_interval = 1000
#store value for these 3 terms so we can plot them later
variational_lower_bound_array = []
log_likelihood_array = []
KL_term_array = []
iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]
for i in range(num_iterations):
    # np.round to make MNIST binary
    #get first batch (200 digits)
    x_batch = np.round(mnist.train.next_batch(200)[0])
    #run our optimizer on our data
    sess.run(optimizer, feed_dict={X: x_batch})
    if (i%recording_interval == 0):
        #every 1K iterations record these values
        vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
        print "Iteration: {}, Loss: {}".format(i, vlb_eval)
        variational_lower_bound_array.append(vlb_eval)
        log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
        KL_term_array.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))

#Visualizing
plt.figure()
#for the number of iterations we had 
#plot these 3 terms
plt.plot(iteration_array, variational_lower_bound_array)
plt.plot(iteration_array, KL_term_array)
plt.plot(iteration_array, log_likelihood_array)
plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
plt.title('Loss per iteration')

#Results
import os
load_model = False
if load_model:
    saver.restore(sess, os.path.join(os.getcwd(), "Trained Bernoulli VAE"))

num_pairs = 10
image_indices = np.random.randint(0, 200, num_pairs)
#Lets plot 10 digits
for pair in range(num_pairs):
    #reshaping to show original test image
    x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
    plt.figure()
    x_image = np.reshape(x, (28,28))
    plt.subplot(121)
    plt.imshow(x_image)
    #reconstructed image, feed the test image to the decoder
    x_reconstruction = reconstruction.eval(feed_dict={X: x})
    #reshape it to 28x28 pixels
    x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
    #plot it!
    plt.subplot(122)
    plt.imshow(x_reconstruction_image)

