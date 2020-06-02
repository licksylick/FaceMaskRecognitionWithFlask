import os
import time
import imutils
from imutils.video import VideoStream

from detect_mask_image import detect
# from detect_mask_video import detect_video
from tensorflow.keras.models import load_model
from detect_mask_video import detect_and_predict_mask

import cv2
from flask import Flask, request, render_template, send_from_directory, url_for, Response
from werkzeug.utils import redirect

__author__ = 'RomanLyskov'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/upload")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images')
    print(target)
    os.makedirs(target, exist_ok=True)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = os.path.join(target, filename)
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        print(destination)

    print("old file: ", filename)
    abs_path = os.path.abspath('images')
    detect(os.path.join(abs_path, filename))
    filename = "img_out.png"
    print(filename)
    return send_from_directory("images", filename, as_attachment=True)

# @app.route('/upload/<filename>')
# def send_image(filename):
#     return send_from_directory("images", filename)


# Real-time face-mask detection
def detect_video(face='face_detector', model='mask_detector.model'):
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face,
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(model)

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video')
def calc():
    return Response(detect_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


# def get_frame():
#     camera_port = 0
#     camera = cv2.VideoCapture(camera_port)  # this makes a web cam object
# =======================================================================================


if __name__ == "__main__":
    app.run(port=4555, debug=True)
