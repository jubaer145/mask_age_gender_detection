import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
def detect_and_predict_age_mask(frame, faceNet, ageNet, maskNet, genderNet, minConf = 0.5):
    # define the list of age buckets our age detector will predict
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]
	genderList = ['Male', 'Female']
    # initialize our results list
	results = []
    # grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	predMasks = []
	predAges = []
	predGenders = []
	locs = []

    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > minConf:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]

            # ensure the face ROI is sufficiently large
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue
            
            # preprocessing for face mask detection
			faceMask = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			faceMask = cv2.resize(faceMask, (224, 224))
			faceMask = img_to_array(faceMask)
			faceMask = preprocess_input(faceMask)
			faceMask = np.expand_dims(faceMask, axis=0)
			predMask = maskNet.predict(faceMask)
			predMasks.append(predMask[0])

			# add the face and bounding boxes to their respective
			# lists
			locs.append((startX, startY, endX, endY))

            # construct a blob from *just* the face ROI
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)
            

			# make predictions on the age and find the age bucket with
			# the largest corresponding probability
			ageNet.setInput(faceBlob)
			predAge = ageNet.forward()
			i = predAge[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = predAge[0][i]
			predAges.append([age, ageConfidence])

			# detecting gender
			genderNet.setInput(faceBlob)
			predGender = genderNet.forward()
			i = predGender[0].argmax()
			gender = genderList[i]
			genderConfidence = predGender[0][i]
			predGenders.append([gender, genderConfidence])
	return predMasks, predAges, predGenders, locs