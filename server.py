import cv2
import time
from flask import Flask, Response

app = Flask(__name__)

# Video capture object (replace with your video file path or camera index)
video_source = 'test-vidoes/explosion.mp4'  # Path to your video file
cap = cv2.VideoCapture(video_source)

# Check if the video source is opened
if not cap.isOpened():
    raise ValueError("Error: Could not open video source.")

# Function to generate frames for the video stream
def generate():
    while True:
        ret, frame = cap.read()

        # If frame is successfully read
        if not ret:
            break

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the frame in the multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Define the route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
