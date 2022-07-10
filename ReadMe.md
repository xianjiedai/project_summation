##1. DATA PREPROCESSING##
  1.1 Read Data
    1.1.1 torchvision.dataset Supported Dataset
      e.g CIFAR 10, MNIST etc.
    1.1.2 Prepared Dataset
      1.1.2.1 Train vs Test
        Split dataset into training dataset and testing dataset
      1.1.2.2 Different Data Format
        CSV, pickle file, png, and etc
  1.2 Data Augmentation
    1.2.1 Rotation
    1.2.2 Random Crop
    1.2.3 Random Noise
    
##2. TRAIN & VALIDATION##
  2.1 Loss
    2.1.1 BCE Loss
    2.1.2 CrossEntropy Loss
    2.1.3 MSE Loss
    2.1.4 Hinge Loss
  2.2 Optimizer
    2.2.1 Zero-order Optimizer
    2.2.2 SGD
      momentum, accelarated, or fixed learning rate ? (tame strong convex step)
    2.2.3 Adam
    2.2.4 Coordinate Descent
      random or importance sampling, steepest random descent
    2.2.5 Second-order Optimizer
      K-FAC
    
  2.3 Scheduler
    dynamic learning rate
  2.4 Weights Initialization
    Xavier, He, sparse initialization
  2.5 Training Details
    epoch, learning rate, and etc

##3. TEST & GENERATE SAMPLES##
