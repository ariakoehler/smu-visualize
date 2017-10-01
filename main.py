from flask import Flask, render_template, Response
import cv2
import sys,os

sys.path.insert(0,os.path.join(os.getcwd(),"./src/"))

from camera import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(cams):
    while True:
        frame = main(cams)
        ret,jpeg = cv2.imencode('.jpeg',frame)
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    theCamera = Cam(0,(320,240))
    theCamera.calibrate()
    #cameraTwo = Cam(2,(320,240))
    #cameraTwo.calibrate()
    return Response(gen([theCamera]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.43.226', debug=True)
