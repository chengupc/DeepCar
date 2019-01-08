#! /usr/bin python
"""
@author chengch 
2019-01-08
由于我们轨道有限，所以数据采集是不连续的，每次只能跑出一小段，然后每次的数据都放在了不同的文件夹里面。
我们需要把所有的数据汇总，而且所有的文件数据都要有一个编号，而不是每个文件夹都有自己的编号。
所以就把所有文件夹下的jpg进行重命名并移动，json文件重命名，还要改变文夹的图像映射，然后再移动。
"""
# -*- coding:UTF-8 -*-

import os
import json
import shutil
from sys import argv

script, sPath = argv

def renameAll(path, num, toPath):
    for file in os.listdir(path):
        if file.startswith("r"):

            temp = file.split("_")[1]
            temp = int(temp.split(".")[0])

            in_json = open(path+file,"r")
            out_json = open(toPath+"record_{}.json".format(temp+num),"w")

            jsonData = json.load(in_json)
            jsonData["cam/image_array"] = "{}_cam-image_array_.jpg".format(temp+num)

            out_json.write(json.dumps(jsonData))

            in_json.close()
            out_json.close()

        elif file.endswith("jpg"):
            temp = int(file.split("_")[0])
            os.rename(path+file,toPath+"{}_cam-image_array_.jpg".format(temp+num))
    return

def getLen(path):
    files = os.listdir(path)
    imgs = []
    for file in files:
        if file.endswith("jpg"):
            imgs.append(file)
    return len(imgs)

def renameDir(path):
    files = os.listdir(path)
    for ind ,file in enumerate(files):
        shutil.move(path+file,path+"{}tub".format(ind+1))

if __name__=="__main__":

    files = os.listdir(sPath)
    pathOne = sPath+files[0]+"/"
    imgNum = getLen(pathOne)

    for file in files[1:]:
        imgNum += getLen(sPath+file+"/")
        renameAll(sPath+file+"/",imgNum, pathOne)
        print("{}".format(file))
    print("ok")
    print("The new file had moved to {}".format(files[0]))
