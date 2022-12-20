import os
import random
import shutil
import csv
import numpy as np
def CopyFile(imageDir,test_rate,save_test_dir,save_train_dir):
    image_number = len(imageDir) 
    test_number = int(image_number * test_rate)
    test_samples = random.sample(imageDir, test_number)
 # copy图像到目标文件夹
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)
        print("save_test_dir has been created successfully!")
    else:
        print("save_test_dir already exited!")
    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
        print("save_train_dir has been created successfully!")
    else:
        print("save_train_dir already exited!")
    for i,j in enumerate(test_samples):
        shutil.copy(test_samples[i], save_test_dir+test_samples[i].split("/")[-1])
    print("test datas has been moved successfully!")
    for train_imgs in imageDir:
        if train_imgs not in test_samples:
            shutil.copy(train_imgs, save_train_dir+train_imgs.split("/")[-1])
    print("train datas has been moved successfully!")
################################
file_path="F:\\A工业公开数据集\\poly"
test_rate = 0.2
################################
file_dirs=os.listdir(file_path)
origion_paths=[]
save_test_dirs=[]
save_train_dirs=[]
for path in file_dirs:
   origion_paths.append(file_path+"/"+path+"/")
   save_train_dirs.append("F:\\A工业公开数据集\\poly_train/"+path+"/")
   save_test_dirs.append("F:\\A工业公开数据集\\poly_test/"+path+"/")
for i,origion_path in enumerate(origion_paths):
    image_list = os.listdir(origion_path)
    image_Dir=[]
    for x,y in enumerate(image_list):
        image_Dir.append (os.path.join(origion_path, y))
    print("%s目录下共有%d张图片！"%(origion_path,len(image_Dir)))
    CopyFile(image_Dir,test_rate,save_test_dirs[i],save_train_dirs[i])
print("all datas has been moved successfully!")

