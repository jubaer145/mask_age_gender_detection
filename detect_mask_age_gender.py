# USAGE
# python detect_mask_video.py

# import the necessary packages
import argparse
import imutils
from imutils.video import VideoStream
import time
import os
from utils import *


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str,
		default="./face_detector",
		help="path to face detector model directory")
	ap.add_argument("-a", "--age", type=str,
		default="./age_detector",
		help="path to age detector model directory")
	ap.add_argument("-m", "--mask", type=str,
		default="./mask_detector.model",
		help = "path to mask detector model directory")
	ap.add_argument("-g", "--gender", type=str,
		default="./gender_detector",
		help = "path to mask detector model directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load our serialized age detector model from disk
	print("[INFO] loading age detector model...")
	prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
	weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
	ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load our serialized gender detector model from disk
	print("[INFO] loading gender detector model...")
	prototxtPath = os.path.sep.join([args["gender"], "gender_deploy.prototxt"])
	weightsPath = os.path.sep.join([args["gender"], "gender_net.caffemodel"])
	genderNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	print("[INFO] loading face mask detector model...")
	maskNet = load_model(args["mask"])

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels

		frame = vs.read()
		frame = imutils.resize(frame, width = 400)

		# detect faces in the frame, and for each face in the frame, 
		# predict age and mask condition

		predMasks, predAges, predGenders, locs = detect_and_predict_age_mask(frame, faceNet, ageNet, maskNet, genderNet, minConf=args["confidence"])

		# loop over the results
		for (box, maskPred, agePred, genderPred) in zip(locs, predMasks, predAges, predGenders):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = maskPred
			# determine the class label and color we'll use to draw
			# the bouding box and text
			maskLabel = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if maskLabel == "Mask" else (0, 0, 255)

			# include the mask probability in the label
			maskLabel = f"{maskLabel}: {max(mask, withoutMask) * 100 : .2f}%"
			mask_y = startY - 10 if startY - 10 > 10 else startY + 10


			ageText = f"Age: {agePred[0]}: {agePred[1] * 100 : .2f}%"
			ageText = ageText if withoutMask > mask else "Can not detect age with mask"
			age_y = startY - 30 if startY - 30 > 30 else startY + 30

			genderText = f"Gender: {genderPred[0]}: {genderPred[1] * 100 : .2f}%"
			# genderText = genderText if withoutMask > mask else "Can not detect gender with mask"
			gender_y = startY - 50 if startY - 50 > 50 else startY + 50

			
			cv2.putText(frame, maskLabel, (startX, mask_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.putText(frame, ageText, (startX, age_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.putText(frame, genderText, (startX, gender_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)			
		
				

			
			
			

			

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break


	cv2.destroyAllWindows()
	vs.stop()


if __name__ == "__main__":
	main()

    