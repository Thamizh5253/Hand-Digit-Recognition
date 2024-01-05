import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist


def model():
    model= tf.keras.models.load_model('handwritten.model')

    loss , accu=model.evaluate(x_test,y_test)

    print(loss  , accu)


    img_no=1
    while os.path.isfile(f"digitsImg1/{img_no}.png"):
     try:
        img = cv2.imread(f"digitsImg1/{img_no}.png")[:,:,0]
        img = np.invert(np.array([img]))
        pred=model.predict(img)
        print(f"This digit is {np.argmax(pred)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.title(f"Predicted digit: {np.argmax(pred)}")
        plt.show()
     except:
        print("Error!")
     finally:
        img_no+=1


model()
