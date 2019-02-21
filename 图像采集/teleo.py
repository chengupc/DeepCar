"""
@artRobot
2019-2-21
Tensorflow小车程序：进行数据采集
    保持匀速行驶，键盘控制小车的角度，每个0.5秒进行一次图片保存和图片对应角度的保存，其中的0.5秒可以自己定义，在timer = threading.Timer(0.5, fun_timer)行
"""
import sys, select, termios, tty
import os
import time
import threading
from ctypes import *
import numpy as np
import cv2
import time

global angleData
global imgInd 
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

def saveImg(path, imgInd): #图像保存函数
    camera = cv2.VideoCapture(0)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))) #尺寸没有做改变，摄像头原始大小
    ret, frame = camera.read()
    cv2.imwrite(path+"{}.png".format(imgInd),frame)

def fun_timer(): #嵌套函数
    global timer 
    
    timer = threading.Timer(0.5, fun_timer) #定义时间间隔
    timer.start()  
    
    angleData.append(getAngle()) #写入角度
    
    saveImg(imgPath, imgInd) 
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
        print(angle)
        return angle
    settings =  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    global settings

if __name__ == "__main__":
    
    vel = 1500
    angle = getAngle()
    
    imgPath = "./img"
    mkDir(imgPath)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    timer = threading.Timer(0.5, fun_timer)
    timer.start()

    while (1):
        pass

data = np.array(angleData)
print "the angle data is saving>>>>>>>>>>> "
np.save("tet.npy",data)
print " the angle data is saving Done"
