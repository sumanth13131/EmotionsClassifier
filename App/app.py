from flask import Flask,render_template,request,url_for,Response
from werkzeug.utils import secure_filename
from detect import getimg
import os
import cv2
app=Flask(__name__)
UPLOAD_FOLDER = './imgssave'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['POST','GET'])
def home():
	if request.method=='POST':
		try:
			for imgs in os.listdir('./imgssave/'):
				f='./imgssave/'+imgs
				os.remove(f)
		except:
			pass

		image = request.files['myfile']
		try:
			filename=secure_filename(image.filename)
			image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			getimg()
			return render_template('show.html')
		except:
			return render_template('index.html')

	return render_template('index.html')

def img_gen():
    while True:
        frame=cv2.imread('./static/detectedimgs/detect.jpg')
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame=jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/img_feed')
def img_feed():
    return Response(img_gen(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True,port=5000)









