import datetime
import os
from io import BytesIO

import cv2
import imutils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, redirect, render_template, request, url_for
from imutils.video import FileVideoStream, WebcamVideoStream
from PIL import Image
from scipy.signal import find_peaks_cwt

matplotlib.use("Agg")
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "assets", "img", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profileCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")


def file_name(filename):
    return os.path.join(UPLOAD_FOLDER, filename)


# ? index route
@app.route("/")
def main():
    return render_template("index.html")


# ? route for uploading image
@app.route("/image")
def image():
    return render_template("image_input.html")


# ? "raw" video feed, returned without wrapping in html template, just used internally
@app.route("/raw_video_feed")
def raw_video_feed():
    return Response(
        generate_next_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ? Actual video live feed with templated html
@app.route("/live_feed")
def live_feed():
    return render_template("live_feed.html")


@app.route("/process_image", methods=["POST"])
def process_image():
    # image_file is the file sent in the HTML form (in the '/image' route)
    image_file = request.files.get("image_file")
    fname, fext = os.path.splitext(image_file.filename)

    raw_image_filename = file_name(f"last_raw_image{fext}")
    generated_image_filename = file_name(f"last_generated_image{fext}")

    image_file.save(raw_image_filename)
    image_data = generate_image(raw_image_filename)

    img = Image.open(BytesIO(image_data))
    # img.show()
    img.save(generated_image_filename)

    return render_template("image_output.html", frame_path=generated_image_filename)


@app.route("/heart_rate")
def heartbeat_input():
    return render_template("heartbeat_input.html")


@app.route("/heartbeat_result", methods=["POST"])
def heartbeat_result():
    # image_file is the file sent in the HTML form (in the '/image' route)
    video_file = request.files.get("heartbeat_video")
    video_file.save(file_name("heart_rate"))

    # connecting with the captured video file taken from mobile
    fvs = FileVideoStream(file_name("heart_rate")).start()

    # getting the number of frames in the video
    no_of_frames = int(fvs.stream.get(7))

    # will eventually contain average red pixel value per frame
    red_avg_values = np.zeros(no_of_frames)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)

    i = 0
    while fvs.more() and i < no_of_frames:
        frame = fvs.read()
        length, width, _ = frame.shape

        # calculating average red channel value in the frame
        red_avg_values[i] = np.sum(frame[:, :, 2]) / (length * width)
        i += 1
    fvs.stop()
    print(f"[INFO] FPS of the video: {fps:.2f}")

    peaks = find_peaks_cwt(red_avg_values, widths=np.ones(red_avg_values.shape) * 2) - 1

    fig = plt.figure()
    plt.xlabel("Frame numbers")
    plt.ylabel("Red channel avg pixel values")
    plt.plot(red_avg_values)
    plt.plot(peaks, red_avg_values[peaks], "x")

    global heart_rate
    heart_rate = 60 * len(peaks) * fps / no_of_frames
    print("heart rate:", heart_rate)

    fig.savefig(file_name("last_generated_plot.png"))

    return render_template(
        "heartbeat_output.html",
        plot_path=file_name("last_generated_plot.png"),
        heart_rate=heart_rate,
    )


# ? detects faces and draws bounding box over faces in the image.
# ? Returns byte-encoded image
def generate_image(filename):
    image = cv2.imread(filename)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ? Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
    )

    profiles = profileCascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) == 0:
        # Draw a rectangle around the face side-profiles in blue colour
        for (x, y, w, h) in profiles:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # encode the image in JPEG format
    (flag, encoded_image) = cv2.imencode(".jpg", image)

    # return output image in byte format
    return bytearray(encoded_image)


# ? takes next frame from camera and detects face in it, and draws bounding box around it.
# ? Returns byte-encoded image (frame)
def generate_next_frame():
    fvs = WebcamVideoStream(0 + cv2.CAP_DSHOW).start()

    # used to record the time when we processed last frame
    prev_frame_time = datetime.datetime.now()

    # used to record the time at which we processed current frame
    new_frame_time = None

    while True:
        frame = fvs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
        )

        profiles = profileCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
        )

        # Draw a rectangle around the faces in green colour
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(faces) == 0:
            # Draw a rectangle around the face side-profiles in blue colour
            for (x, y, w, h) in profiles:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ? calculate and diplay real time FPS on screen
        new_frame_time = datetime.datetime.now()
        curr_fps = 1 / (new_frame_time - prev_frame_time).total_seconds()
        prev_frame_time = new_frame_time
        curr_fps = int(curr_fps)

        cv2.putText(
            frame,
            f"Current FPS: {curr_fps}",
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # encode the frame in JPEG format
        (flag, encoded_image) = cv2.imencode(".jpg", frame)

        # yield the output frame in the byte format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n"
        )
    fvs.stop()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
