import numpy as np
import tensorflow as tf
import os,cv2
interpreter = tf.lite.Interpreter(model_path="NewModel.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
classes=['anger','disgust','happy','neutral','sad','surprise']
def load_image(img):
    img=cv2.resize(img,(128,128))
    data =img
    return data/255.0
prototxtPath = "./detector/deploy.prototxt"
weightsPath ="./detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)




def getimg():
	f=[]
	for file in os.listdir('./imgssave/'):
		f.append(file)
	path='./imgssave/'+f[0]
	img_o =cv2.imread(path)
	img_o=cv2.resize(img_o, (400,400))
	(h, w) = img_o.shape[:2]
	blob = cv2.dnn.blobFromImage(img_o, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	for i in range(0,detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence >= 0.50:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0,startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			if (abs(startX-endX))>50  and abs(startY-endY)>50:
				image=img_o[startY-15:endY+20, startX-40:endX+40]
				image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				data=[load_image(image)]
				data=np.array(data,dtype=np.float32).reshape(interpreter.get_input_details()[0]['shape'])
				interpreter.set_tensor(input_index,data)
				interpreter.invoke()
				j=interpreter.get_tensor(output_index)
				i=np.argmax(j)
				font = cv2.FONT_HERSHEY_SIMPLEX
				org = (startX-20, startY-20)
				fontScale = 0.6
				color = (0,0, 255)
				thickness = 1
				cv2.rectangle(img_o, (startX-40, startY-15), (endX+40,endY+20), (0, 0, 255), 3)
				cv2.putText(img_o,classes[i] , org, font,fontScale, color, thickness, cv2.LINE_AA)
				cv2.putText(img_o,classes[0]+f': {int(round(j[0][0],2)*100)} %',(5,15), font, 0.5, (0,255,0),1,cv2.LINE_AA)
				cv2.putText(img_o,classes[1]+f': {int(round(j[0][1],2)*100)} %',(5,30), font, 0.5, (0,255,0),1,cv2.LINE_AA)
				cv2.putText(img_o,classes[2]+f': {int(round(j[0][2],2)*100)} %',(5,45), font, 0.5, (0,255,0),1,cv2.LINE_AA)
				cv2.putText(img_o,classes[3]+f': {int(round(j[0][3],2)*100)} %',(5,60), font, 0.5, (0,255,0),1,cv2.LINE_AA)
				cv2.putText(img_o,classes[4]+f': {int(round(j[0][4],2)*100)} %',(5,75), font, 0.5, (0,255,0),1,cv2.LINE_AA)
				cv2.putText(img_o,classes[5]+f': {int(round(j[0][5],2)*100)} %',(5,90), font, 0.5, (0,255,0),1,cv2.LINE_AA)
	try:
		for i in os.listdir('./static/detectedimgs/'):
			file='./static/detectedimgs/'+i
			os.remove(file)
	except:
		pass
	cv2.imwrite('./static/detectedimgs/detect.jpg', img_o)








