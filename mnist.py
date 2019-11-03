#!/usr/bin/env python
# coding: utf-8

# # CNN implementation for recognition of MNIST hand written digit dataset

# <b> Importing required libraries </b>

# In[37]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split


# <b> Importing data </b>

# In[43]:


data = pd.read_csv(r'C:\Users\Ankit\Desktop\mnist.csv')


# In[44]:


data.shape


# <b>Visualizing one of the row</b>

# In[45]:


num = data.iloc[10,1:].values
num=num.reshape(28,28).astype('uint8')
plt.imshow(num)


# <b> For all the rows converting pandas series to numpy array of 28*28 dimension and then creating X's (features) and y (label) </b><br>
# <b>keras.utils.to_categorical is just like one hot encoding on the labels</b>

# In[53]:


df_x = data.iloc[:,1:].values.reshape(len(data),28,28,1)
y = data.iloc[:,0]  
print(y.value_counts())  #  to get the distribution of all the labels
y = data.iloc[:,0].values # series to numpy array
df_y = keras.utils.to_categorical(y,num_classes=10) # creating categories corressponding to every class


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2) # 80/20 split of data into train and test


# In[55]:


model = Sequential()  # CNNs are sequencial model
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))  # 32 3*3 filters, channel = filter and we have only one filter which is in the last
model.add(MaxPooling2D(pool_size=(2,2)))  # 2*2 Max pooling
model.add(Flatten())
model.add(Dense(100)) # 100 neurons in the fully connected NN
model.add(Dropout(0.5)) # to avoid overfitting
model.add(Dense(10)) # output layer of NN
model.add(Activation('softmax'))  # to get percentage
model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])


# In[31]:


model.summary()   # total parameters to learn = 542230


# In[56]:


model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=5)


# In[57]:


model.evaluate(x_test,y_test)


# In[ ]:




