#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import test
import pandas as pd
import numpy as np
import csv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


def Distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# In[3]:


def file_read(txts):
    reading = []
    for idx,f_txt in enumerate(txts):
        while True:
            line = f_txt.readline()
            if not line: break
            file_name, gt = line.rstrip().split('\t')
            file_name = f'0{idx+1}/'+(file_name.split('/'))[-1]
            reading.append([file_name,gt])
    reading.sort(key = lambda x : x[0])
    return np.array(reading).T


# In[4]:


def pred_append(ret,image_path,predict,sp):
    for path,pred in zip(image_path,predict):
        path = sp+(path.split('/'))[-1]
        ret.append([path,pred])
    ret.sort(key = lambda x : x[0])
    return ret


# In[5]:


img_path1,pred1,time1 = test.demoPy(image_folder='../datasets/test/01/',saved_model='./model01.pth', imgW=64, imgH = 32, character="0123456789.t")


# In[6]:


img_path2,pred2,time2 = test.demoPy(image_folder='../datasets/test/02/',saved_model='./model02.pth', imgW=128, imgH = 32, character="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# In[7]:


img_path3,pred3,time3 = test.demoPy(image_folder='../datasets/test/03/',saved_model='./model03.pth', imgW=100, imgH = 32, character="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -")


# In[8]:


txts = []
txts.append(open('../datasets/test/01/gt_test_01.txt','r'))
txts.append(open('../datasets/test/02/gt_test_02.txt','r'))
txts.append(open('../datasets/test/03/gt_test_03.txt','r'))

gts = file_read(txts)
gt_DF = pd.DataFrame({"path" : gts[0], "GT" : gts[1]})
for txt in txts:
    txt.close()


# In[9]:


preds = pred_append([],img_path1,pred1,'01/')
preds = pred_append(preds,img_path2,pred2,'02/')
preds = pred_append(preds,img_path3,pred3,'03/')
pred_count = len(preds)
preds = np.array(preds).T
pred_DF = pd.DataFrame({"path" : preds[0], "pred":preds[1]})

gt_pred = pd.merge(gt_DF,pred_DF,on="path",how="outer")
gt_pred = gt_pred.replace(np.nan,'X').astype('string')


# In[10]:


f_csv = open('result.csv','w',newline='')
f_txt = open('GTX.txt','w')
wr = csv.writer(f_csv)

f_txt.write(f'{"IMG_PATH":20s}\t{"GT":15s}\t{"pred"}\n{"-"*55}\n')
wr.writerow(['IMG_PATH','GT',"Pred","Accuracy"])

cal_avg_acc = [0,0]

for idx,row in gt_pred.iterrows():
    if row['pred']=='X' or row['GT']=='X':
        f_txt.write(f'{row["path"]:20s}\t{row["GT"]:15s}\t{row["pred"]}\n')
    else:
        cache1 = Distance(row['GT'],row['pred'])
        cache2 = len(row['GT']+row['pred'])/2
        cal_avg_acc[0] += cache1
        cal_avg_acc[1] += cache2
        wr.writerow([row['path'],row['GT'],row['pred'],f'{round(100*(1-cache1/cache2),2)}%'])

wr.writerow(['','','Average Accuracy',f'{round(100*(1-cal_avg_acc[0]/cal_avg_acc[1]),2)}%'])
wr.writerow(['Total Time',f'{round(time1+time2+time3,3)}s','Time per Image',f'{round((time1+time2+time3)/pred_count*1000,3)}ms'])

f_csv.close()
f_txt.close()




