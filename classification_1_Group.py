# for modeling
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import asarray
from numpy import savetxt

from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

x_train = np.load("./Xtrain_Classification_Part1.npy")
y_train = np.load("./Ytrain_Classification_Part1.npy")
x_test = np.load("./Xtest_Classification_Part1.npy") 

x_train = x_train.reshape(-1, 50,50, 1)
x_test = x_test.reshape(-1, 50,50, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

train_Y_one_hot = to_categorical(y_train)

train_X,valid_X,train_label,valid_label = train_test_split(x_train, train_Y_one_hot, test_size=0.1, random_state=130)
batch_size = 64
epochs = 65
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(50,50,1),padding='same'))

model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
                
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
                
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7 ,restore_best_weights=True)

model_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,callbacks=[callback],verbose=1,validation_data=(valid_X, valid_label))

y_pred = model.predict(x_test)

lista =[]

for i in range(len(y_pred)):
    lista.append(np.argmax(y_pred[i]))
    
np.save("y_predictions",lista) #saving the the predictions of y into a .npy file

