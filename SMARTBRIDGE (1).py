#!/usr/bin/env python
# coding: utf-8

# # AI ENABLED WEED RECOGNITION SYSTEM

# In[ ]:


#IMPORTING THE LIBRARIES


# In[38]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[ ]:


#INITIALISING THE MODEL 


# In[39]:


model=Sequential()


# In[ ]:


#ADDING CONVOLUTION LAYER


# In[40]:


model.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))


# In[ ]:


#ADDING MAXPOOLING LAYER


# In[41]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[ ]:


#ADDING FLATTEN LAYER


# In[42]:


model.add(Flatten())


# In[ ]:


#ADDING HIDDEN LAYER TO CNN


# In[43]:


model.add(Dense(output_dim=50,activation="relu",init="random_uniform"))


# In[ ]:


#ADDING ANOTHER HIDDEN LAYER TO GET THE MODEL TRAINED EFFICIENTLY


# In[44]:


model.add(Dense(output_dim=20,activation="relu",init="random_uniform"))


# In[ ]:


#ADDING OUTPUT LAYER TO CNN


# In[45]:


model.add(Dense(output_dim=4,activation="softmax",init="random_uniform"))


# In[ ]:


#ADDING VARIOUS DATA PROCESSING TECHNIQUES


# In[46]:


from keras.preprocessing.image import ImageDataGenerator


# In[47]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


#IMPORTING THE IMAGE DATASET


# In[48]:


x_train=train_datagen.flow_from_directory(r"E:\dataset\train_dataset",target_size=(64,64),batch_size=32,class_mode="categorical")
x_test=test_datagen.flow_from_directory(r"E:\dataset\test_dataset",target_size=(64,64),batch_size=32,class_mode="categorical")


# In[49]:


print(x_train.class_indices)


# In[ ]:


#COMPILING THE MODEL


# In[50]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


#TRAINING THE MODEL WITH EPOCHS=25


# In[51]:


model.fit_generator(x_train,steps_per_epoch=50,epochs=25,validation_data=x_test,validation_steps=16)


# In[ ]:


#SAVING THE MODEL


# In[52]:


model.save("smartbridge1.h5")


# In[ ]:


#LOADIND THE MODEL FOR PREDICTION


# In[53]:


from keras.models import load_model


# In[54]:


from keras.preprocessing import image


# In[56]:


model=load_model("smartbridge1.h5")


# In[57]:


img=image.load_img(r"E:\dataset\test_dataset\grass\395.tif",target_size=(64,64))


# In[58]:


x=image.img_to_array(img)


# In[59]:


import numpy as np


# In[60]:


x=np.expand_dims(x,axis=0)


# In[61]:


res=model.predict_classes(x)


# In[75]:


if(res[0]==0):print("It is Broad Leaf")


# In[77]:


if(res[0]==1):print("It is Grass")


# In[79]:


if(res[0]==2):print("It is soil")


# In[81]:


if(res[0]==3):print("It is soybean")


# # open cv for video processing

# In[ ]:


#import cv library


# In[1]:


import cv2


# In[ ]:


#argument required to capture video from a camera


# In[2]:


cap = cv2.VideoCapture(0)


# In[ ]:


#Define the codec and create VideoWriter object


# In[3]:


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


# In[4]:


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break
    else:
        break


# In[ ]:


# Release everything if job is finished


# In[5]:


cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




