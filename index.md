Machine Learning
-----------------------
**Why Kernel is useful?**

Machine learning techniques try to classify samples from different sets by looking for separators in space to separate them. However, in reality, samples cannot always be separable. It is something Kernel could help. Kernels map samples from low dimension to high dimension where the low-dimensional samples can be separated after being mapped to higher dimensional space.

The mapping/expanding process can be thought as deriving new features from old features. For instance, a sample [x1, x2] can be mapped to high dimension space [x1^2, x2^2, x1x2, x2x1]. If we can simply do this to any training samples and testing samples, then problem solved! Why do we need kernel? 

It turns out that doing this brutal force feature expanding is very computational expansive. Therefore, we need to come up with a smarter solution and that is kernel. 

**How it is achieved?**

The guy who invented kernel found a very important pattern of any machine-learning algorithm. That is they all deal with the testing sample by doing dot product. Here is an example:

![alt tag](./Machine Learning/1.png)

Where Wt is the separator in the Perceptron Algorithm while Xin is the training sample. The way it classifies the testing sample is by simply dot product the new samples with all its components:

![alt tag](./Machine Learning/2.png)

See the interaction between the training data and the testing data? It dot products them.

The guy who invented kernel also found out that even expanding both training and testing sample to high dimensional space, the dot product result of them can be expressed in a *really simple* form. This way the computational cost is reduced while we can still enjoy the separability from high dimensional space.

What are the simple forms of the dot products depends on what kernel we use:

![alt tag](./Machine Learning/3.png)


Embedded System
----



Kernel
-------


