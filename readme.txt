Convolutional Neural Networks for Gesture Recognition
Dataset: 42 sets of letters from A-Z drawn 5 times, measured through accelerometer and gyroscope of mobile devices
Goal: To product a Convolutional Neural Network to predict from a 26-class output, (letters A-Z) using time series data gathered by 42 participants.

Summary of features: 
- 3 convolutional layers of sizes 158->312->520 
	- kernel size 10 
	- padding 2 
	- stride 1 (default) 
	- max pool = 2 
	- dilation = 1 (default) 
- 3 Fully Connected Layers of 208->104->26 proceeding Conv. Layers
- ReLu Activation between every layer
- Logarithmic Softmax

Hyperparameters
- Learning Rate = 0.005175
- Batch Size = 32
- Epochs ~ 100 (dependent on speed of covergence in training accuracy)
- Adam Optimizer (Default settings, as described in PyTorch Documentation)
- Loss Function = Cross Entropy

