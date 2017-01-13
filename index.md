
#Notes

##Machine Learning

###Kernel

**Why Kernel Method is useful?**

Machine learning techniques try to classify samples from different sets by looking for separators in space to separate them. However, in reality, samples cannot always be separable. It is something Kernel could help. Kernels map samples from low dimension to high dimension where the low-dimensional samples can be separated after being mapped to higher dimensional space.

The mapping/expanding process can be thought as deriving new features from old features. For instance, a sample [x1, x2] can be mapped to high dimension feature space [x1^2, x2^2, x1x2, x2x1]. This process is called feature map, where we usually use Phi(x) to represent the feature map function. If we can simply do this to any training samples and testing samples, then problem solved! Why do we need kernel? 

It turns out that doing this brutal force feature expanding is very computational expansive. Therefore, we need to come up with a smarter solution and that is kernel. 

**How it is achieved?**

The guy who invented kernel found a very important pattern of any machine-learning algorithm. That is they all (at least SVM and Perceptron) deal with the testing sample by doing dot product. Here is an example:

![alt tag](./Machine Learning/1.png)

Where Wt is the separator in the Perceptron Algorithm while Xin is the training sample. The way it classifies the testing sample is by simply dot product the new samples with all its components:

![alt tag](./Machine Learning/2.png)

See the interaction between the training data and the testing data? It dot products them.

The guy who invented kernel also found out that even expanding both training and testing sample to high dimensional space, the dot product result of them can be expressed in a *really simple* form. This way the computational cost is reduced while we can still enjoy the separability from high dimensional feature space. So we avoid working in the high demensional feature space explicitly. 

What are the simple forms of the dot products depends on what kernel we use:

![alt tag](./Machine Learning/3.png)

There is a nice example of applying kernel method to perceptrons. <https://en.wikipedia.org/wiki/Kernel_perceptron>

###Perceptron
Perceptron uses two steps of computation to classify samples as one of the two labels. First, it does a dot product between linear seperator and input vector. Second it uses a threshold to test if the result from the dot production and output a binary value as the label. Here is an example of a two-demension perceptron with a threshold b: Sign(W1*X1 + W2*X2 - b).

Perceptron can be thought of as a single-layer Neural Network.

A perceptron:

![alt tag](./Machine Learning/5.png)

**How to train a perceptron**

In the online learning situation, where we get training data sequentially, the misclassified instance will contribute to the seperator by dragging it towards the direction of that misclassified sample a bit. Therefore next time the mistake is less likely to happen by a bit more. 

Here is the code:

```
%Perceptron algorithm
%   w0 is the initial weight vector (d * 1)
%   X is feature values of training examples (d * n)
%   Y is labels of training examples (1 * n)

for j = 1 : n
	y_hat(j) = perceptron_pred(w , X(:,j));
		if(y_hat(j) ~= Y(j))
				w = w+Y(j)*X(:,j); % drag by adding the misclassified label to the weights vector
		end
end
```
**How to kernelize a perceptron**

Kernelized perceptron stores a counter vector that keep track of the times Xi is misclassified; and the set of training data as a trained model. The set of training data is stored in the form of a kernel matrix K as a lookup table. The kernel matrix should be constructed such that if X1 ∈ R d×n and X2 ∈ R d×m, then K ∈ R n×m. Note that each element of the K matrix is computed according to the kernel function. The polynomial kernel of degree-3 with an offset of 1 is defined as: 

![alt tag](./Machine Learning/5_1.png)

So if we want to compute k(Xi, Xj), we can simply index K(i,j). Usually one of the X1 or X2 should be training data while the other is testing data. But during the training, we can use training data as both X1 and X2. In this case, the K matrix is just another form of representing training data. 

Here is the code:
```
%Kernel perceptron algorithm
%   a is the count vector (1 * n)
%   X is feature values of training examples (d * n)
%   Y is labels of training examples (1 * n)

for i = 1 : n
        y_hat(i) = kernel_perceptron_pred(a, Y, K, i);
        if y_hat(i) ~= Y(i)
            a(i) = a(i)+1;
        end
end
```

where in kernel_perceptron_pred, we have:
```
%PERCEPTRON_PRED: Make prediction using Gram matrix of kernel,
%				past labels and counts, and index i.
%   a is counting vector (1 * n)
%   Y is labels of training examples (1 * n)
%	K is the Gram matrix such that K(i, j) = Kernel(X_i, X_j) 
%   i is the index of current observation

K_i = K(i,:);
pred = sign(sum((a.*Y)*K_i')); % i is the ith test sampe K_i is K(Xi, X[training samples]) 
```

###Naive Bayes

Chain rule is the basis of Bayes therom while Naive Bayes make Bayes therom more practical in real-world problem.

Compared with logistic regression, Naive Bayes is generative, which is derived from the joint distributtion of Y and X P(Y, X), instead of merely the probability distribution of Y given X P(Y | X). Therefore, it requires more training data to achieve good result.

Naive Bayes approach is a generative model since it tells a story of how the data is generated (both P(Y) and P(X|Y). But discrimitive model such as logistic regression how tells the current state of the data (P(Y|X)).


###Regression

The goal of linear regression is to find the W (weights) vector to minimize the square error (also called loss function). It is mathmatically the same thing with maximizing the likelihood of the conditional likelihood L(w; x,y), which is the summarization of evidence between data and unknown parameters (w). The second way is also how the logistic regression is trained, so it is in some sense more general between linear regression and logistic regreession.

![alt tag](./Machine Learning/7.png)

The way to find w that maximize/minimize an particular expression (i.e. loss function, likelihood function) is to use gradient descent. Here is an example of training logistic regression using gradient descent:
```
for k = 1:max_iter
    [f,g] = fv_grad( w, X, y );
    w = w + step * g;
	eps = abs((f - f_prev) / f_prev);
	    if eps <= stop_criteria
	        break;
	    end
end
```

where fv_grad returns the objective function value f and the gradient g w.r.t w at point w_curr:
```
tmp = -log(1+exp(w_curr'*X)) + y.*(w_curr'*X);
f = -sum(tmp,2); % f is negative log likelihood
g = (y - 1./(1+exp(-w_curr'*X)))*X'; 
g = g';
```

###Support Vector Machine

Unlike the training process for previous algorithm which try to maximize the possibility of observing the outcome of the training samples or minimizing the error, Support Vector Machine directly optimize for the maximum margin separator. 

![alt tag](./Machine Learning/16.png)

Divide both side by gamma, we change the optimization to:

![alt tag](./Machine Learning/17.png)

Where W' = W/gamma.

To allow errors we replace the inequality constraints with:

![alt tag](./Machine Learning/12.png)

where 􏰗i are slack variables that allow an example to be in the margin (1 > 􏰗Yi > 0, also called a margin error) or misclassified (􏰗Yi > 1). Since an example is misclassified if the value of its slack variable is greater than 1, the sum of the slack variables is a bound on the number of misclassified examples. Minimizing ||w||2 can be augmented with a term CΣY􏰗i to penalize misclassification and margin errors. The optimization problem now becomes:

![alt tag](./Machine Learning/13.png)

Using the method of Lagrange multipliers, the dual formulation of this optimization problem becomes:

![alt tag](./Machine Learning/14.png)

This way, the final classifier is: ![alt tag](./Machine Learning/15.png) and The points Xi for which ai ≠ 0 are called the *support vector*.

[Assignments](./assignments/hw5.pdf)


###Boosting

First of all, boosting is a general method for improving the accuracy of any given learning algorithm. In general, boosting tries to find rules of thumb (highly accurate rule) by calling the weak learner repeatedly on cleverly chosen datasets:

![alt tag](./Machine Learning/20.png)

Note that rules are different from features. 

**Adaptive boosting**


###Sample Complexity, PAC and VC Dimension
Sample complexity guarantees quantify how many training samples we need to see from the underlying data distribution D, inorder to guarantee that for all hypotheses in the class of target function we have their true error closer to their training error. This is important because we can only optimize for the training error but it is the generalization error (true error, the error if we apply the trained model to future data) that we really care about. The PAC and sample complexity also tells us how much data we need in order to get training error close to the generalization error with high probability:
![alt tag](./Machine Learning/18.png)

Sample complexity of a machine learning algorithm is the number of training samples needed for the algorithm to successfully learn a target function with small error of both training error and true error.
![alt tag](./Machine Learning/19.png)

VC-Dimension provides a measure of the complexity of a "hypothesis space" which is your hypothesis of the data distributtion in the problem you want to solve. VC bounds are one kind of sample complexity guarantee, where the bound depends on the VC-dimension of the hypothesis class and they are particularly useful when the class of functions is infinite. 

The target function is generated from the hypothesis space. For instance, if your hypothesis is that the regression curve will be 2nd order polynomial, then one target function could be y = X^2 + 1. However, your hypothesis is your assumption about the truth and it decides how complex you want your model to be.


### Clustering
given a set S = {X1, .. Xn} of n points in d-dimensional space, the goal of k-means clustering is to find a set of centers c1, . . . , ck that minimize the k-means objective:
![alt tag](./Machine Learning/21.png)

Note that min() ensure that we only sum up the distance of each point when it is assigned to the nearest center.

Brutal force all the partition posibilities can guarantee the optimal centers with minimal k-means objective, however the running time is exponential of the number of data points. Lloyd's Method is efficient in practice and often outputs reasonably good clusterings. It interates between two steps: improving partitioning and improving the centers. i.e., (i) improving the partitioning C1, . . . , Ck by reassigning each point to the cluster with the nearest center, and (ii) improving the centers c1, . . . , ck by setting ci to be the mean of those points in the set Ci for i = 1, ..., k. Note that Ci is cluster (i.e., points set) while ci is the center of a cluster. So we have pseudocode here: 
![alt tag](./Machine Learning/22.png)

As we can tell, this method relys on how many clusters we assume and where the initial centers locates. For picking the number of clusters, we can run a k sweep and pick up an "elbow" point, since increasing the number of centers beyond this point results in relatively little gain. 

Intuitively we want each center to locate on each clusters with no multiple centers sharing the same cluster. For picking the initial locations of the centers, we can do it randomly or we can do k-means++ initialization approach where the initial centers are more likely to be one center per cluster.

### Bayesian Network
Bayes Network is a graphical model that represents a set of random variables and their conditional dependencies via a directed acyclic graph. D-seperation is a useful tool to decide wether two nodes or two groups of nodes are conditional dependent, conditioning on another node or another group of nodes.

### Hidden Markov Models
HMM is a special kind of Bayesian Network. In terms of the notion of d-separation, state nodes St-1 and St+1 are conditionally independent given St. Observation nodes Ot-1 and Ot+1 are conditionally independent given Ot. In fact, in the HMM graphical model, d-separation reduces to the common notion of separation in graphs. We have that any two subsets of nodes A and B are conditionally independent of a third one C, when the nodes in C and edges with endpoints in C are removed. 

Given a fixed HMM, i.e., a HMM with fixed parameters, Theta, Theta(start), Theta(stop), and Gamma, there are various queries that we may want to answer. For example, we could want to know what is the probability of a sequence of observations O1:T., PO1:T(O1:T). Evaluating this requires marginalizing over all posible sequences of states that may have generated O1:T, i.e.,

![alt tag](./Machine Learning/23.png)

However, doing the summation naively is intractable. An efficient approach will exploit the HMM structure: given a sequence of observations O1:T, what is the most likely sequence of states S1:T that gave rise to O1:T. This is called MAP or Viterbi decoding and it is written as: 
 
![alt tag](./Machine Learning/24.png)

We can do Viterbi decoding naively which has N to the squre of T time complexity, or we can derive an efficient algo. to do it.

### Markov blanket
The Markov blanket MB(v) of node v belongs to V is the set of parents, co-parents and children of v. Given v's MB(v), v is d-seperated from the rest of the nodes.


### Reinforcement Learning
What we do will affect what we see next.
Not only about prediction, but also about making decision.

Always want to maximize reward, but prefer ealier reward than later. Therefore, in the reward function, future rewards have less weights than current rewards. (weight decay)

Value function for each policy. Our goal is to find the policy that maximizes the value function.

Function to be learned is PI: S->A
The training instances are <<S,A>, R> S:state A:action R:reward

Supprisingly, the optimized policy for statei is also the optimized policy for statej.


### Markov Assumption
The probability of most recent state only depends on the last state and the action after the last state.

###MISC

**What is linear function**

A linear function is just multiply between matrixes, the result of which is a polinomial function of degree zero or one. It can always be represented by W . X. For instance, aX + b but not aX^2 + b. We can see linear function all over the place in machine learning algorithms in the form of WX + b where W is the weights, X is the data, and b is the bias. W and b are trained.

**Understand overfit**

Bias and Variance are used to evaluate an algorithm (our hypothesis about the data distribution in the dataset):

Bias: How accurate the model is to predict current dataset, when using this algorithm.
Variance: How various the models are when apply the same algorithm across different datasets.

We see an algorithm with small bias but big variance, and that is overfit.

## Neural Network

### Backpropagation
Backpropagation is a common methond for training a neural netowrk. The two phases include 1. propagtion and 2. weight update. And repeat phase 1 and phase 2 until the performace of the network is satisfactory (optimize the weights). 


1. Forward propagation of a training pattern's input through the neural network in order to generate the propagation's output activations.

2. Calculate the total error and the partial derivative of Etotal with respect to all weights (how much a change in Wi affects the total error.) This is done by applying chain rule of the the gradient. (<http://cs231n.github.io/optimization-2/#backprop>)

3. We substract the derivatives times by certain ratio from the current weights and use the updated weights for next iteration.

This ratio (percentage) influences the speed and quality of learning; it is called the learning rate. The greater the ratio, the faster the neuron trains; the lower the ratio, the more accurate the training is. The sign of the gradient of a weight indicates where the error is increasing, this is why the weight must be updated in the opposite direction.

A nice animation: <https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/>

Deeper network is more expressive.

Parallelism: Neural network involves a lot of matrix multiplitation which can be accelerated by modern hardware. This makes computation for deeper network no longer computationally imposible. There exist a lot of tools (e.g., tensorflow) takes neural network training as a sequence of matrix operations.

Use softmax and cross-entropy loss to improve (accelerate, more stable) deep neural network.

###Convolutional Neural Network or (CNN)
CNN is a type of neural network however, each note is an image (matrix) instead of a value, while each weight is a matrix instead of a value. The convolutions between Weights and inputs compute the output, rater that the multiplications. By making explicit assumption that the inputs are images, CNN encodes certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

CNN basically chains up convolutions and combines them by downsampling.

Deeplearning abstract useful features from raw images which reduces number of features comparing with using raw images as features.

Here is a nice tutorial with an animation of CNN (<http://cs231n.github.io/convolutional-networks/>)


##Deep Learning 

### Network

Let's first look at a boolean (i.e., 0, 1) output neural network that shows 1) the good property of sigmoid 2) reducing error by calculating error-weighted derivatives in the back propagations, and adding them to the weights to update.

http://iamtrask.github.io/2015/07/12/basic-python-network/

In the above example the output is boolean. However, in a more realistic case where we have multiple labels, we can use softmax regression together with cross-entropy entropy function (loss funtion). 

####Softmax (an output layer representation)
Good thing about the softmax regression is it gives us a list of values between 0 and 1 that add up to 1 (probabilities of each label add up to 1). So the output layer is softmax.
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

Here is a nice illustration of this architecture:
![alt tag](./Deep Learning/d7.png)

where xj (first column) is the pixels of an image, Wi,j is the weight of computing pixel j being part of label i. All the evidance tallies are then fed into a softmax layer to compute the possibilities.

Here are the following computations:
![alt tag](./Deep Learning/d8.png)

Note that there is no activation layer in this example. Usually, there should be an activation layer with sigmoid computation before softmax to 1) normalize the output from wx+b and 2) add nonlinearity to the process. see figure:
![alt tag](./Deep Learning/d10.png)

####Cross-entropy entropy function (a loss function representation)

It is the same as negative log-likelihood where we try to minimize, or a multi-class classification problem.
The funtion is:
![alt tag](./Deep Learning/d9.png)
Where the y' is the true distribution (the one-hot vector with the digit labels). One-hot means the vector has only one '1' which happens at the correct digit label's position. y is our predicted probability distribution. In some rough sense, the cross-entropy is measuring how inefficient our predictions are for describing the truth. Here are details about cross-entropy: http://colah.github.io/posts/2015-09-Visual-Information/

####Gradient descent algorithm (a procedure to minimize loss)
Stochastic gradient descent is a stochastic approximation of the gradient descent but requires less training time. From wikipedia: "To economize on the computational cost at every iteration, stochastic gradient descent samples a subset (i.e. dividing the full set of data into minibatches) of summand functions at every step. This is very effective in the case of large-scale machine learning problems." 
More specifically, we update weights and biases based on the averaged nabla (differentials) computed from *one mini_batch* of the training data set, rather than the entire set of the training data. This increases the number of iterations in one epoch (one epoch is defined as one round of parameter updating where all training data points are used exactly once.)

###Convolutional Neural Network

The only difference between a CNN and a normal Neural Network are the convolutional and pooling layers.

example of 2d matrix convolution: http://www.songho.ca/dsp/convolution/convolution2d_example.html

Convolutional neural networks alternate between convolutional and pooling layer 
![alt tag](./Deep Learning/d1.png)

Why pooling: 
1) we cain robustness to extract spatial location of features (i.e. make detection robust to the exact location of the eye (see figure below)):
![alt tag](./Deep Learning/d2.png)

2) lower the translation variances for more stable result (or introducing invariance):
![alt tag](./Deep Learning/d3.png)

Why need rectification? It is also for more stable result with invariance. For, eg. gain stability by lossing information such as whether an edge is black-to-white or white-to-black:
![alt tag](./Deep Learning/d5.png)

Simple fact of convolutional network:
![alt tag](./Deep Learning/d4.png)


Dropout, Momentum and Batch Normalization have significant improvement on CNN.

Breakthrough work using DCNN on image classification: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Based on the experiment on AlexNet, the depth of the DCNN is the key to achieve high accuracies.


### Restricted Boltzmann Machine
This network for unsupervised learning and is good for abstracting underlying structure/relationship, or in other word finding patterns of the input. Finding this low-dimensional representation of the input is useful to remove the redundancy and overfitting of the input data. Therefore its hidden layer is usually used for features for other machine learning algorithms. Using the hidden layer from RBM instead of using raw pixels/signals as features makes the classifier more robust to subtal pixel/signal shifting or rotations, which are usually invisible to human eyes but can hurt machine learning models significantly. This process is so called "Pretraining". 

As the same as other machine learning models, RBM uses gradient descent to minimize the loss function, which in this case is the negative log-likelihood. By utilizing the "energy function" borrowed from physics, the data negative log-likelihood gradient then can be expressed as a particularly interesting form:

![alt tag](./Deep Learning/d11.png)

Notice that the above gradient contains two terms, which are referred to as the positive and negative phase. The terms positive and negative do not refer to the sign of each term in the equation, but rather reflect their effect on the probability density defined by the model. The first term increases the probability of training data (by reducing the corresponding free energy), while the second term decreases the probability of samples generated by the model (see the figure below). (This is borrowed from online tutorial: http://deeplearning.net/tutorial/rbm.html#contrastive-divergence-cd-k)

![alt tag](./Deep Learning/d12.png)

To approximate the loss function (negative log-likelihood), we use Contrastive Divergence (CD) with Gibbs Sampling (k steps), usually denoted as CD-k. 

![alt tag](./Deep Learning/d14.png)
![alt tag](./Deep Learning/d13.png)

#### Gibbs Sampling
Gibbs sampling is applicable when the joint distribution is not known explicitly or is difficult to sample from directly, but the conditional distribution of each variable is known and is easy (or at least, easier) to sample from. The Gibbs sampling algorithm generates an instance from the distribution of each variable in turn, conditional on the current values of the other variables. It can be shown (see, for example, Gelman et al. 1995) that the sequence of samples constitutes a Markov chain, and the stationary distribution of that Markov chain is just the sought-after joint distribution.

![alt tag](./Deep Learning/d15.png)

A comprehensive RBM implementation can be found here: https://github.com/yusugomori/DeepLearning/blob/master/python/RBM.py

### Sparse Coding 
Sparse coding can be used to extract features:

http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding

ZCA preprocessing is usually used to pre-process the data to remove the "obvious" structure from the data that might pollute the learning result. After ZCA normalization, the mean is 0 and covariance is the identity (so called whitening in CV).

## TensorFlow
TensorFlow is a Deep Learning library developed by Google. To setup TensorFlow: https://www.tensorflow.org

In the demo code below, a convolutional network with two hidden layers is used to train the MNIST dataset. In TensorFlow some basic data structures are used such as session, placeholder, and tensor. Session is a lot like a function, which defines a sequence of operations inside it. We can pass variables to a session using feed_dict(arg1:value1, arg2:value2, ...). placeholders defined in the session are the arguments to which we will need to pass values during the training. Some common placeholders are training data, training labels, and dropout rate. Tensor is a unique data typed defined in TensorFlow, which basically is a nd array.

TensorFlow provides some useful tools to handle data, such as queue and coordinators. They are useful for getting training instances out of the dataset. 

```
with tf.Session() as sess:

  x = tf.placeholder("float", shape=[None, 784])
  y_ = tf.placeholder("float", shape=[None, 10])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  sess.run(tf.initialize_all_variables())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(20000):

    # actual training 
    train_image_batch = tf.to_float(train_image_batch)
    
    tx,ty = sess.run([train_image_batch, train_label_batch])
    
    tx = tx/255. # input has to be normalized to 0 - 1 

    train_step.run(feed_dict={x: tx, y_: ty, keep_prob: 0.5})

    # probe the training accuracy
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:tx, y_: ty, keep_prob: 1.0})
      print "step %d, training accuracy %g"%(i, train_accuracy)

    if i%100 == 0: # probe the validation accuracy
      test_image_batch = tf.to_float(test_image_batch)
      vx,vy = sess.run([test_image_batch, test_label_batch])
      vx = vx/255.0
      
      test_accuracy = accuracy.eval(feed_dict = {x:vx, y_:vy, keep_prob:1.})
      print "step %d, validation accuracy %g" % (i, test_accuracy)
      
       
  coord.request_stop()
  coord.join(threads)
  sess.close()
```


##Analog

###Setup dc bias for aplifier
Voltage follower biasing compared with voltage divider biasing is more power-efficient, can be used to power many circuits (the same as voltage divider biasing), the affect to other part of the circuit is infinitesimal.

###Bypass Capacitor

**where to put it:** As close to the IC's power input as possible. Extra distance increase the additional series inductance which lowers the bandwidth of frequencies the capacitor can bypass. In wide bandwidth circuits, the amount of series inductance sets an upper bound on the ability of the bypass circuit to provide a low impedance for the power supply pin.

**Different capacitors:** Ceramic capacitors are the most common capacitor type since they are inexpensive, offer a wide range of values, and provide
solid performance. 
Tantalum, OSCON, and Aluminum Electrolytic capacitors are all polarized (specifically to be used as a bypass capacitor). 

Tantalum found their niche in low-voltage systems.

Aluminum electrolytic capacitors are a common choice for lowto-medium frequency systems, but not switching circuits (they
hold their charge too well which doesn’t suit them for the rapid
cycling of production testing). 

OSCON is a special capacitor type developed to provide low parasitics, wide frequency range and full temperature range (the best quality available for the highest price tag). If you have the budget, these capacitors will provide quality bypass for any circuit.

For a good practice we use three capacitors with different value to bypass wideband of noise. Note that the values always differ by the level of magnitude of 10 and also we chose smaller package size for smaller capacitors. If the same package was used for each of the capacitors,
their high frequency responses will the same. Effectively, this negates the use of the smaller capacitors. See figure: 
![alt tag](./Analog/1.png)

Here is a good tutorial of [how to choose bypass fileter](./Analog/bypass_cap.pdf)

###Reduce EMI Noise on PCB
SuperSensor pcb digital and analog seperation. Will put more information when I have time.

###Antenna Impedance Match
SuperSensor antenna 50 ohm impedance match. Will put more information when I have time.

### Eagle 

snap-on-grid-sch.ulp
find.ulp
zoom_unrounted.ulp

##Others

###Kappa statistic
Classifiers built and evaluated on data sets of different class distributions can be compared more reliably through the kappa statistic. Using kappa statistic, we allow models of different class distributions to be more easily compared.
<http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english>


### Java
####Interface VS. Abstract Classes
Key differences are 1) a class can extend only one abstract class while it can implement multiple interfaces, and 2) abstract class can define concrete method while interface can only define abstract method.

```
public class Main extends animal implements livingthingsA, livingthingsB{ // a class can extend only one abstract class while it can implement multiple interfaces
	int value;
	
	public Main(int v){
		this.value = v;
	}

	@Override
	public int eat(int a) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void breath() {
		// TODO Auto-generated method stub
		
	}
}

abstract class animal{
	public void fart(){ // abstract class can define concrete method 
		System.out.println("fart");
	}
	abstract public int eat(int a); // and abstract method
}

interface livingthingsA {
	abstract public void breath(); // interface can only define abstract method
}

interface livingthingsB {
	abstract public void breath(); 
}
```



###Python
####Generators 
Generator in python returns an iterator which returned results follow the order of the "yield" statement in the generator function. For example:

```
import random
def lottery():
    for i in xrange(6):
        yield 1
    yield 2
for random_number in lottery():
    print "And the next number is... %d" % random_number
```

#### List comprehension 
Useful trick to create a list out of another list:
```
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()
lenList = [len(word) for word in words if word != "the"]
print lenList
```

#### Multiple function argument
**options in the following code means we are gonna send functions arguments by keyword. So the order does not matter.

```
def bar(first,second,thuid, **options):
    if options.get("action") == "sum":
        return first+second+third
    if options.get("action") == "first":
        return first

print bar(1,2,3,action = "first")
```

#### Serialization
Serialization is the process of turning an object in memory into a stream of bytes so you can do stuff like store it on disk or send it over the network. Pickle module offers a powerful serialization/deserializaiton tool for python objects.

```
import pickle
a = {'hello': 'world'}
with open('filename.pickle', 'wb') as handle:
  pickle.dump(a, handle)
with open('filename.pickle', 'rb') as handle:
  b = pickle.load(handle)
```

#### Code introspection
Introspection is an act of self examination. In computer programming, introspection is the ability to determine the type of an object at runtime.

The dir() function returns a sorted list of attributes and methods belonging to an object.
The type() function returns the type of an object.
More of these functions can be found in http://zetcode.com/lang/python/introspection/


#### Numpy array index trick
Use comparison between two numpy arrays to gemerate an index.

```
import numpy as np
a = np.random.rand(10)
b = np.ones(10) - 0.5
c = np.zeros(10)
c[b>a] = 1
print c
```

### Git
#### SSH
Setup ssh on a laptop: https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/

How SSH works:

The SSH protocol employs a client-server model to authenticate two parties and encrypt the data between them using symmetric, asymmetric encryption and hashes. A nice tutorial can be found: https://www.digitalocean.com/community/tutorials/understanding-the-ssh-encryption-and-connection-process#symmetric-encryption,-asymmetric-encryption,-and-hashes
