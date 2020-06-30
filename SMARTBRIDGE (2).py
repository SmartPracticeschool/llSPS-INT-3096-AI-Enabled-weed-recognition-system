#!/usr/bin/env python
# coding: utf-8

# # AI ENABLED WEED RECOGNITION SYSTEM

# In[1]:


#IMPORTING THE LIBRARIES


# In[2]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[3]:


#INITIALISING THE MODEL 


# In[4]:


model=Sequential()


# In[5]:


#ADDING CONVOLUTION LAYER


# In[6]:


model.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))


# In[7]:


#ADDING MAXPOOLING LAYER


# In[8]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


#ADDING FLATTEN LAYER


# In[10]:


model.add(Flatten())


# In[11]:


#ADDING HIDDEN LAYER TO CNN


# In[12]:


model.add(Dense(output_dim=50,activation="relu",init="random_uniform"))


# In[13]:


#ADDING ANOTHER HIDDEN LAYER TO GET THE MODEL TRAINED EFFICIENTLY


# In[14]:


model.add(Dense(output_dim=20,activation="relu",init="random_uniform"))


# In[15]:


#ADDING OUTPUT LAYER TO CNN


# In[16]:


model.add(Dense(output_dim=4,activation="softmax",init="random_uniform"))


# In[17]:


#ADDING VARIOUS DATA PROCESSING TECHNIQUES


# In[18]:


from keras.preprocessing.image import ImageDataGenerator


# In[19]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


# In[20]:


#IMPORTING THE IMAGE DATASET


# In[21]:


x_train=train_datagen.flow_from_directory(r"E:\dataset\train_dataset",target_size=(64,64),batch_size=32,class_mode="categorical")
x_test=test_datagen.flow_from_directory(r"E:\dataset\test_dataset",target_size=(64,64),batch_size=32,class_mode="categorical")


# In[22]:


print(x_train.class_indices)


# In[23]:


#COMPILING THE MODEL


# In[24]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[25]:


#TRAINING THE MODEL WITH EPOCHS=25


# In[26]:


model.fit_generator(x_train,steps_per_epoch=50,epochs=25,validation_data=x_test,validation_steps=16)


# In[ ]:


#SAVING THE MODEL 


# In[27]:


model.save("ai.h5")


# In[ ]:


#LOADING THE SAVE MODEL FOR PREDICTION


# In[28]:


from keras.models import load_model


# In[29]:


from keras.preprocessing import image


# In[30]:


model=load_model("ai.h5")


# In[31]:


img=image.load_img(r"C:\Users\Sai prasanna Kumar\OneDrive\Desktop\405.tif",target_size=(64,64))


# In[32]:


x=image.img_to_array(img)


# In[33]:


import numpy as np


# In[34]:


x=np.expand_dims(x,axis=0)


# In[35]:


res=model.predict_classes(x)


# In[36]:


res  #FINAL RESULT OR PREDICTION


# In[37]:


if(res[0]==0):print("It is a BroadLeaf")


# In[38]:


if(res[0]==1):print("It is Grass")


# In[39]:


if(res[0]==2):print("It is Soil")


# In[40]:


if(res[0]==3):print("It is soybean")


# # OPEN CV FOR IMAGE PROCESSING

# In[41]:


import cv2


# In[42]:


import numpy as np


# In[ ]:


#creating an object to capture video(i.e in this case webcam)


# In[43]:


cap = cv2.VideoCapture(0)


# In[44]:


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[45]:


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[46]:


print(gray)


# In[ ]:




