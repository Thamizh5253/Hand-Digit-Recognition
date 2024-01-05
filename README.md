# Overview
This project aims to recognize handwritten digits using a deep learning model trained on the MNIST dataset. The project consists of two main parts:

## 1)Model Building:
In this phase, a neural network model is built using TensorFlow and Keras. The model is trained on the MNIST dataset, which is a collection of 28x28 pixel grayscale images of handwritten digits (0-9).

## 2)Testing and Deployment: 

After training the model, it is saved and deployed to recognize handwritten digits from images. A separate script is provided to load the saved model and test it on user-provided images.

## Requirements
**Libraries and Installations:**
    
📌**TensorFlow**

📌**NumPy**

📌**OpenCV (cv2)**

📌**Matplotlib**

You can install these libraries using pip:

```bash
  pip install tensorflow numpy opencv-python matplotlib
```


## Files
The project consists of two main Python files:

## model.py: 
This file contains the code for building, training, and saving the handwritten digit recognition model.
## main.py: 
This file contains the code for loading the saved model and testing it on user-provided images.
## Usage
### Building the Model 
📌Run the model.py script to build and train the model on the MNIST dataset.

📌 The trained model will be saved as handwritten.model in the current directory.
### Testing and Deployment
📌Draw a digit (0-9) on a canvas of size 28x28 pixels using any drawing tool like MS Paint.

📌Save the image as a PNG file in the digitsImg1 directory with a filename like 1.png, 2.png, etc.

📌Run the model.py script to load the saved model and test it on the handwritten digit images saved in the digitsImg1 directory.

📌The script will display the predicted digit along with the image for each saved image.

## Model Architecture
The neural network model used for this project consists of the following layers:

**Input Layer:**
Flatten layer to reshape the input images to a 1D array (28x28 = 784 pixels).

**Hidden Layer 1:**
 Dense layer with 128 neurons and ReLU activation function.
Hidden Layer 2: Dense layer with 128 neurons and ReLU activation function.

**Output Layer:** Dense layer with 10 neurons (corresponding to digits 0-9) and softmax activation function for multi-class classification.

## Notes
📌Ensure that the images saved in the digitsImg1 directory are 28x28 pixels grayscale images.

📌The model's performance can vary based on the quality and clarity of the handwritten digits in the images.

📌By following the instructions above, you can build, test, and deploy a handwritten digit recognition model using deep learning techniques.