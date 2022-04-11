import argparse
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required =True)
ap.add_argument("-c","--confidence",type = float,default = 0.5)
ap.add_argument("-t","--threshold",type = float,default = 0.3)
args = vars(ap.parse_args())
#load the coco names labels to our yolo model on which they are trained 
labelspath = 'coco.names'
LABELS = open(labelspath).read().strip().split("\n")
#initialize the list of colors to represent each possible class label
COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype = "uint8")
#get path to yolo weights and yolo config file 
weightspath = "yolov3.weights"
configpath = "yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configpath,weightspath)
cap = cv2.VideoCapture(args["video"])
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
while(cap.isOpened()):
    for i in range(1,fps):
      ret,image = cap.read()
    if(ret==True):
        (H,W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #construct a blob from input image and then perform a forward pass of the yolo object detector 
        blob = cv2.dnn.blobFromImage(image, 1/255.0,(416,416),swapRB = True,crop = False)
        net.setInput(blob)
        layeroutputs = net.forward(ln)
        boxes=[]
        confidences=[]
        classIDs = []
        for output in layeroutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if (confidence > args["confidence"])and(LABELS[classID] == "person"):
                    box = detection[0:4] * np.array([W,H,W,H])
                    (centerx,centery,width,height) = box.astype("int")
                    x = int(centerx - (width/2))
                    y = int(centery - (height/2))
                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        #apply non-maxima suppression to suppress the weak overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes,confidences,args["confidence"],args["threshold"])
        if(len(idxs) > 0):
            #lop over the indexes which we are keeping 
            for i in idxs.flatten():
                (x,y) = (boxes[i][0],boxes[i][1])
                (w,h) = (boxes[i][2],boxes[i][3])
                #draw the bounding box rectangle and label on the image 
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                text = "{}:{:.4f}:{}".format(LABELS[classIDs[i]],confidences[i],(w*h))
                cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
        cv2.imshow("Image",image)
        if(cv2.waitKey(25)&(0xFF == ord('q'))):
            break
    else:
        break   
















