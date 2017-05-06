import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

# 0 is the MacBook Pro front camera
# 1 is the external webcam
video_capture = cv2.VideoCapture(int(sys.argv[2]))

while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
	gray,
	# 1.1 default
	scaleFactor=1.3,
	# 5 default
	minNeighbors=5,
	# 30x30 default
	minSize=(70, 70),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	# "Ye old way"
	# faces = faceCascade.detectMultiScale(
	# gray,
	# # 1.1 default
	# scaleFactor=1.1,
	# # 5 default
	# minNeighbors=5,
	# # 30x30 default
	# minSize=(30, 30),
	# flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	# )

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
