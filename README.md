# Framework Summary #
## 1. DATA PREPROCESSING ##
### 1.1 Read Data ###
- torchvision.dataset Supported Dataset

      e.g CIFAR 10, MNIST etc.
      
- Prepared Dataset

Train vs Test: split dataset into training dataset and testing dataset

Different Data Format: CSV, pickle file, png, and etc
        
### 1.2 Data Augmentation
- Rotation
- Random Crop
- Random Noise
    
## 2. TRAIN & VALIDATION ##
### 2.1 Loss
- BCE Loss
- CrossEntropy Loss
- MSE Loss
- Hinge Loss
 ### 2.2 Optimizer
- Zero-order Optimizer
- SGD
      momentum, accelarated, or fixed learning rate ? (tame strong convex step)
      
- Adam
- Coordinate Descent
      random or importance sampling, steepest random descent
- Second-order Optimizer
      K-FAC   
### 2.3 Scheduler
dynamic learning rate
### 2.4 Weights Initialization
Xavier, He, sparse initialization
### 2.5 Training Details
epoch, learning rate, and etc

## 3. TEST & GENERATE SAMPLES ##
