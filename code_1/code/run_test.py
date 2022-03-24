################ Interface #################

result_csv_location="."
test_set_location="../datasets/test"
csv_file_name="result.csv"

#########################################


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import time
from PIL import Image
import random

model2=tf.keras.applications.EfficientNetB0(weights=None,input_tensor=tf.keras.Input((492,654,3)),classes=2)
model2.load_weights("./b0")

empty_set=os.listdir(test_set_location+"/0")
loaded_set=os.listdir(test_set_location+"/1")
empty_num=len(os.listdir(test_set_location+"/0"))
loaded_num=len(os.listdir(test_set_location+"/1"))
GT=[0 if(i<empty_num) else 1 for i in range(empty_num+loaded_num)]
prediction=[]

start=time.time()
temp=empty_num
while(temp>0):
    x=[]
    for i in range(min(300,temp)):
        img=np.array(Image.open(test_set_location+"/0"+"/"+empty_set[i]))
        x.append(img)

    x=np.asarray(x)
    x=tf.convert_to_tensor(x)
    x=tf.cast(x,float)
    #x=tf.image.crop_and_resize(x,[[0,0.01,1,0.99] for i in range(min(300,temp))],list(range(min(300,temp))),[492,654])
    x=tf.image.crop_and_resize(x,[[0,0.12,1,0.88] for i in range(min(300,temp))],list(range(min(300,temp))),[492,654])
    #print(model2.predict(x))
    prediction.append(tf.argmax(model2.predict(x),1))
    temp-=300
    empty_set=empty_set[300:]
    print("empty set "+str(max(temp,0))+" left")

temp=loaded_num
while(temp>0):
    x=[]
    for i in range(min(300,temp)):
        img=np.array(Image.open(test_set_location+"/1"+"/"+loaded_set[i]))
        x.append(img)

    x=np.asarray(x)
    x=tf.convert_to_tensor(x)
    x=tf.cast(x,float)
    #x=tf.image.crop_and_resize(x,[[0,0.01,1,0.99] for i in range(min(300,temp))],list(range(min(300,temp))),[492,654])
    x=tf.image.crop_and_resize(x,[[0,0.12,1,0.88] for i in range(min(300,temp))],list(range(min(300,temp))),[492,654])
    #print(model2.predict(x))
    prediction.append(tf.argmax(model2.predict(x),1))
    temp-=300
    loaded_set=loaded_set[300:]
    print("loaded set "+str(max(temp,0))+" left")


prediction=tf.concat(prediction,0)

inference_speed=(time.time()-start)/(empty_num+loaded_num)
accuracy=[bool(prediction[i]==GT[i]) for i in range(empty_num+loaded_num)]
average_accuracy=sum(accuracy)/(empty_num+loaded_num)


f=open(result_csv_location+"/"+csv_file_name, "w")
f.write("Image list,GT,Prediction,accuracy\n")
empty_set=os.listdir(test_set_location+"/0")
loaded_set=os.listdir(test_set_location+"/1")

for i in range(empty_num):
    f.write(empty_set[i]+",0,"+str(int(prediction[i]))+","+str(accuracy[i])+"\n")
for i in range(loaded_num):
    f.write(loaded_set[i]+",0,"+str(int(prediction[i+empty_num]))+","+str(accuracy[i+empty_num])+"\n")
    
f.write("Average Accuracy,"+str(average_accuracy)+"\n")
f.write("Inference Speed (ms),"+str(inference_speed*1000)+"ms/image\n")

