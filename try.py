import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from keras import layers,callbacks,utils, applications,optimizers
from keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
files = os.listdir( "D://Codes/project pro/Face_Sketch Database/erchive")
files
img_array= []
label_array = []
path = "D://Codes/project pro/Face_Sketch Database/erchive"
for i in range(len(files)):
    file_sub = os.listdir(path+"/"+files[i])
    for k in tqdm(range(len(file_sub))):
        try:
            img = cv2.imread(path+files[i]+"/"+file_sub[k])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(96,96))
            img_array.append(img)
            label_array.append(i)
        except:
            pass
import gc
gc.collect()

img_array = np.array(img_array)/255.0
label_array = np.array(label_array)

X_train,X_test,Y_train,Y_test=train_test_split(img_array,label_array,test_size=0.15)

model=Sequential()
pretrained_model=tf.keras.applications.EfficientNetB0(input_shape=(96,96,3), 
                                                      include_top=False,
                                                     weights="imagenet")
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(1))
model.summary()
model.compile(optimizer="adam",loss="mean_squared_error",metrics=["mae"])
ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                   monitor="val_mae",
                                                   mode="auto",
                                                   save_best_only=True,
                                                   save_weights_only=True)
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                             monitor="val_mae",
                                             mode="auto",
                                             cooldown=0,
                                             patience=5,
                                             verbose=1,
                                             min_lr=1e-6)
Epoch=300
Batch_Size=64
history=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),
                  batch_size=Batch_Size,
                  epochs=Epoch,
                  callbacks=[model_checkpoint,reduce_lr])
# Run
model.load_weights(ckp_path)
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()
# save model
with open("model.tflite","wb") as f:
    f.write(tflite_model)
    prediction_val=model.predict(X_test,batch_size=64)
prediction_val[:20]
X_test[:20]