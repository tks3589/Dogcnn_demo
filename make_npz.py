from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
import os, cv2, numpy as np, pandas as pd

path = "dataset/" 
label_cnt=-1
imgs=[]
label=[]
labelname=[]
for dirspath in os.listdir(path):
        fullpath = join(path, dirspath) 

        if isdir(fullpath): 
                if dirspath not in labelname: 
                        label_cnt+=1
                        labelname.append(dirspath) 
                
                for imgname in os.listdir(fullpath):
                        imgpath = join(fullpath, imgname)                   
                        img = cv2.resize(cv2.imread(imgpath), (64,64), interpolation=cv2.INTER_CUBIC)                       
                        imgs.append(img)
                        label.append(label_cnt)
                        
                        
np.savez('labelname.npz', labelname=labelname)
np.savez('dataset.npz', data=imgs, label=label)