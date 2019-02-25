# -*- coding: UTF-8 -*-
"""
@artRobot
2019-2-21
实时采集图像数据以及角度数

"""
import sys, select, termios, tty
import os
import time
import threading
from ctypes import *
import numpy as np
import cv2 as cv
import time

global angleData

imgInd = 0

angleData = []

def mkDir(path): #创建需要保存图像的文件夹
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print "----- new folder .....----"
        print "----- OK ---------"
    else:
        print "------------ There is this folde! ----- "

def getImg(cap, imgInd, path):
    ret, frame = cap.read()
    frame = cv.resize(frame,(int(frame.shape[1] / 10), int(frame.shape[0] / 10)))
    size = frame.shape
    cv.imwrite(path+"/{}.jpg".format(imgInd), frame)
    imgInd += 1

def fun_timer(): #嵌套函数
    global timer 
    global imgInd
    global cap
    global imgPath

    timer = threading.Timer(0.5, fun_timer) #定义时间间隔
    timer.start()  
    
    angleData.append(getAngle()) #写入角度
    
    getImg(cap, imgInd, imgPath)
    
    imgInd += 1
    
    print(getAngle())

def getKey(): #获取键盘值函数
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key
angle = 1500
def getAngle():
    global angle
    settings = termios.tcgetattr(sys.stdin)
    while (1):
        key = getKey()
        if key == 'a' and (angle <= 2500):
            angle += 50
            #rint(angle)
        elif key == 'd' and (angle >= 500):
            angle -= 50
            #print(angle)
        elif (key == '\x03'):   #for ctrl + c exit
            data = np.array(angleData)
            np.save("tet.npy",data)
            break
        #print(angle)
        return angle
    settings =  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    global settings

if __name__ == "__main__":
    
    vel = 1500
    angle = getAngle()
    cap = cv.VideoCapture(0)

    lib_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/lib" + "/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    
    imgPath = "./img"
    
    try:
        car = "/dev/ttyUSB0"
        if (lib.art_racecar_init(38400, car) < 0):
            raise
            pass
        timer = threading.Timer(0.5, fun_timer)
        timer.start()
    
        while (1):
            pass

    except:
        print "error"

    finally:
        cap.release()
        print"finally"

data = np.array(angleData)
print "the angle data is saving>>>>>>>>>>> "
np.save("tet.npy",data)
print " the angle data is saving Done"
