import cv2
import requests
import json
import sys
import time
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the endpoint URL and API key
prediction_endpoint = os.getenv("PREDICTION_ENDPOINT")
api_key = os.getenv("PREDICTION_KEY")

# # Define the endpoint URL and API key
# prediction_endpoint = 'https://cctvthreatdetection-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/df22819a-4cca-4292-9b6b-e246a1c6d0eb/classify/iterations/Iteration3/image'
# api_key = 'CkHoO2aVUvcQarDoT6g7pM5E5fUXtmZ0sZI22cC0JBS9qZHj2APbJQQJ99ALACYeBjFXJ3w3AAAIACOGMIb8'

# Path to your input and output video files
video_path = 'test-vidoes/explosion.mp4'
output_video_path = 'output_video_with_predictions.mp4'

# Set up the headers for authentication
headers = {
    'Prediction-Key': api_key,
    'Content-Type': 'application/octet-stream'
}

# Open the input video file
cap = cv2.VideoCapture(video_path)

# Check if video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames in the video: {total_frames}")

# Get the video frame width and height for creating the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # 30 FPS

# Define a function for the spinner
def spinner():
    chars = ['|', '/', '-', '\\']
    while True:
        for char in chars:
            yield char

# Create a spinner generator
spin = spinner()

# Loop to read frames from the video
frame_count = 0
start_time = time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was read successfully, ret will be True
    if not ret:
        print("End of video or error reading frame.")
        break

    # Increment the frame counter
    frame_count += 1

    # Convert frame to JPEG (binary data) format
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Send the frame (as an image) to the prediction API
    response = requests.post(prediction_endpoint, headers=headers, data=img_bytes)

    # Show a spinner in the terminal while waiting for the API response
    sys.stdout.write(f'\rProcessing frame {frame_count}/{total_frames}... {next(spin)}')
    sys.stdout.flush()

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
        print(f"Request failed for frame {frame_count} with status code {response.status_code}")
        print(response.text)

    # Write the modified frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\nOutput video saved at {output_video_path}. Total time taken: {total_time:.2f} seconds.")
