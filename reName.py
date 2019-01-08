#! /usr/bin python

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
