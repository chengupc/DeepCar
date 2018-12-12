"""
@chengch
for video to imge by opencv
2018-12-12
"""

import cv2

vc = cv2.VideoCapture('/home/parallels/Desktop/cai/cai.mp4') #read in the video

c = 1 # the first image's name

if vc.isOpened(): # Determine whether it can be opened properly
    rval, frame = vc.read()
else:
    rval = False

timeF = 15 # time interval 

while rval: 
    rval, frame = vc.read()
    if ( c % timeF == 0):
        cv2.imwrite("/home/parallels/Desktop/cai/"+str(c)+".jpg", frame)
    c = c+1
    cv2.waitKey(1)
vc.release()

