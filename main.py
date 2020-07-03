'''
YOLOv3 Object Detection
'''

import cv2
import numpy as np 

#Storing Label Names of each object
labelsFile = 'coco.names'
labels = []
with open(labelsFile,'rt') as f:
	labels = f.read().rstrip('\n').split('\n')

#Loading YOLOv3-tiny configuration file and weights 
modelConfig = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

#Initializing network
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)

#Setting backend as OpenCV
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

#Setting target processor as CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Initializing camera
cap = cv2.VideoCapture(0)
WIDTH = 320
HEIGHT = 320

#Setting confidence and non-maximum suppression threshold
confThreshold = 0.7
nmsThreshold = 0.2

def detectObjects(outputs, img):
	
	height, width, channels = img.shape
	
	#Initialzing lists for bounding box coordinates, class IDs and cconfidence
	boxList = []
	classIds = []
	confidenceList = []

	#Looping over each output
	for output in outputs:

		for det in output:
			#Extracting prediction value
			scores = det[:5]
			#Extracting classId as maximum index
			classId = np.argmax(scores)
			#Extracting confidence
			confidence = scores[classId]

			#Filtering weak predictions 
			if (confidence > confThreshold):

				#Grabbing dimensions for bounding box
				w, h = int(det[2] * width), int(det[3] * height)
				x, y = int((det[0] * width) - w/2), int((det[1] * height) - h/2)
				
				#Adding dimensions, classId and confidence value
				boxList.append([x, y, w, h])
				classIds.append(classId)
				confidenceList.append(float(confidence))

	#Applying Non Maximum Suppression 
	indices = cv2.dnn.NMSBoxes(boxList, confidenceList, 
		confThreshold, nmsThreshold)

	for index in indices:
		
		#Extracting coordinates of primary bounding box
		index = index[0]
		box = boxList[index]
		(x, y, w, h) = box[0], box[1], box[2], box[3]

		#Drawing rectangle over detected object
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255) ,2)

		#Adding label
		cv2.putText(img, f'{labels[classIds[index]].upper()} {int(confidenceList[index]*100)}%',
			(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

while True:

	ret, img = cap.read()

	#Converting input image to blob to pass thorugh network
	blob = cv2.dnn.blobFromImage(img, 1/255, (WIDTH, HEIGHT),
		[0, 0, 0], 1, crop = False)

	#Passing blob through network
	net.setInput(blob)

	#Grabbing layer names from Network
	layerNames = net.getLayerNames()

	#Extracting first element from list of layers
	outputNames = [layerNames[i[0]-1] 
		for i in net.getUnconnectedOutLayers()]

	#Grabbing output layers with predictions 
	#outputs = (centreX, centreY, width, height, confidence)
	outputs = net.forward(outputNames)
	
	#Passing outputs and image frame through detectObjects
	detectObjects(outputs, img)

	cv2.imshow('Feed', img)
	
	#Escape key to quit 
	key = cv2.waitKey(1)
	if(key == 27):
		break

cap.release()
cv2.destroyAllWindows()

