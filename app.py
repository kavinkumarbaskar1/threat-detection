import cv2
import requests
import time
import os
from dotenv import load_dotenv
import gradio as gr
import numpy as np

# Load environment variables from the .env file
load_dotenv()

# Define the endpoint URL and API key
prediction_endpoint = os.getenv("PREDICTION_ENDPOINT")
api_key = os.getenv("PREDICTION_KEY")

# Path to your input and output video files
video_path = 'test-vidoes/explosion.mp4'
output_video_path = 'output_video_with_predictions.mp4'

# Set up the headers for authentication
headers = {
    'Prediction-Key': api_key,
    'Content-Type': 'application/octet-stream'
}

# Define a function for the spinner
def spinner():
    chars = ['|', '/', '-', '\\']
    while True:
        for char in chars:
            yield char

# Check if the input source is a streaming video or a file
def get_video_source(source='stream'):
    if source == 'stream':
        # Try to open a webcam or stream
        cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
        if not cap.isOpened():
            print("Error: Could not open streaming video source.")
            return None
        print("Streaming video source detected.")
    else:
        # Open the video file as fallback
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None
        print("Video file detected.")
    return cap

# Initialize frame count globally
frame_count = 0

# Open the video capture object (streaming or video file)
cap = get_video_source(source='file')  # Change to 'file' if you want to use the file as fallback

if cap is None:
    print("Exiting program: No video source found.")
    exit()

# Get total number of frames in the video (for file input)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
print(f"Total number of frames (if file): {total_frames}")

# Get the video frame width and height for creating the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # 30 FPS

# Create a spinner generator
spin = spinner()

# Function to process and show live video frames with predictions
def process_video_frame():
    global frame_count  # Use global frame_count

    ret, frame = cap.read()
    if not ret:
        return None, None  # Return None if the frame is invalid

    frame_count += 1  # Increment frame_count

    # Convert frame to JPEG (binary data) format
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Send the frame (as an image) to the prediction API
    response = requests.post(prediction_endpoint, headers=headers, data=img_bytes)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        prediction_data = response.json()

        # Extract the predictions and sort them by probability (descending)
        predictions = prediction_data.get('predictions', [])
        if predictions:
            # Sort predictions by probability in descending order
            sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
            highest_prob_prediction = sorted_predictions[0]
            tag_name = highest_prob_prediction['tagName']
            probability = highest_prob_prediction['probability']

            # Draw the tag name with the highest probability on the frame
            label = f"{tag_name}: {probability * 100:.2f}%"
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

    # Convert both the original and processed frames to RGB for display in Gradio
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frame_rgb = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)  # Read the original frame
    
    return original_frame_rgb, frame_rgb  # Return both original and processed frames

# Gradio interface for displaying original and processed video side by side
def video_interface():
    return gr.Interface(fn=process_video_frame, inputs=None, outputs=["image", "image"], live=True)

# Function to process video and display loader for static video
def process_static_video():
    # Set up loading spinner
    time.sleep(1)  # Just for showing the loader for a brief moment (this can be adjusted)
    
    # Create a cap object to read the video file
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame, show loader, and store results in output frames
    processed_frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to JPEG (binary data) format
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        # Send the frame (as an image) to the prediction API
        response = requests.post(prediction_endpoint, headers=headers, data=img_bytes)

        if response.status_code == 200:
            # Parse the JSON response
            prediction_data = response.json()

            # Extract the predictions and sort them by probability (descending)
            predictions = prediction_data.get('predictions', [])
            if predictions:
                # Sort predictions by probability in descending order
                sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
                highest_prob_prediction = sorted_predictions[0]
                tag_name = highest_prob_prediction['tagName']
                probability = highest_prob_prediction['probability']

                # Draw the tag name with the highest probability on the frame
                label = f"{tag_name}: {probability * 100:.2f}%"
                cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Append the processed frame to the list
        processed_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Return the list of processed frames (will be displayed in Gradio)
    return processed_frames, processed_frames

# Gradio interface for video processing with static video and loading state
def static_video_interface():
    return gr.Interface(fn=process_static_video, inputs=None, outputs=["gallery", "gallery"], live=False)

# Create Gradio interface for streaming video or static video processing
def create_gradio_interface(video_type='stream'):
    if video_type == 'stream':
        # Live video stream interface (original vs processed side by side)
        return gr.Interface(fn=process_video_frame, inputs=None, outputs=["image", "image"], live=True)
    else:
        # Static video processing interface (show loader during processing)
        return static_video_interface()

# Launch Gradio interface for video stream (or static video)
interface = create_gradio_interface(video_type='file')  # Change to 'stream' for live streaming

interface.launch(share=True)
