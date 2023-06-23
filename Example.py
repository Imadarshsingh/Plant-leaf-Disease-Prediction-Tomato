import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model(r"C:\Users\91892\Desktop\Plant-Leaf-Disease-Prediction\model.h5")
print(model)

print("Model Loaded Successfully")

tomato_plant = r"C:\Users\91892\Desktop\Plant-Leaf-Disease-Prediction\Dataset\val\Tomato - Target_Spot\Tomato___Target_Spot (1).JPG"
test_image = tf.keras.preprocessing.image.load_img(tomato_plant, target_size = (128, 128)) # load image 
  
test_image = tf.keras.preprocessing.image.img_to_array(test_image)/255 # convert image to np array and normalize
test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
result = model.predict(test_image) # predict diseased plant or not
  
pred = np.argmax(result, axis=1)
print(pred)
if pred==0:
    print( "Tomato - Bacteria Spot Disease")
       
elif pred==1:
    print("Tomato - Early Blight Disease")
        
elif pred==2:
    print("Tomato - Healthy and Fresh")
        
elif pred==6:
    print("Tomato - Late Blight Disease")
       
elif pred==4:
    print("Tomato - Leaf Mold Disease")
        
elif pred==5:
    print("Tomato - Septoria Leaf Spot Disease")
        
elif pred==3:
    print("Tomato - Target Spot Disease")
        
elif pred==7:
      print("Tomato - Tomoato Yellow Leaf Curl Virus Disease")

elif pred==8:
      print("Tomato - Tomato Mosaic Virus Disease")

elif pred==9:
      print("Tomato - Two Spotted Spider Mite Disease")