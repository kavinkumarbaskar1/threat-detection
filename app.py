import cv2
import requests
import json
import sys
import time
import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables from the .env file
load_dotenv()

# Define the endpoint URL and API key
prediction_endpoint = st.secrets.PREDICTION_ENDPOINT
api_key = st.secrets.PREDICTION_KEY

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

# Streamlit UI
st.title('Video Processing with Predictions')
st.write('Upload a video, and the app will process each frame to add predictions.')

if "upload_button" not in st.session_state:
    st.session_state.upload_button = False

if "live_button" not in st.session_state:
    st.session_state.live_button = False

# Handle video upload
if st.session_state.upload_button:
    st.subheader("Upload a Video File for Processing")
    # Video upload widget
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Temporary file storage for uploaded video
        temp_input_path = tempfile.mktemp(suffix=".mp4")
        with open(temp_input_path, 'wb') as f:
            f.write(video_file.read())

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_input_path)

        # Check if video file was opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            st.stop()

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Temporary file for output video
        temp_output_path = tempfile.mktemp(suffix="_output.mp4")

        # Set up the video writer to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output .mp4 file
        out = cv2.VideoWriter(temp_output_path, fourcc, 30.0, (frame_width, frame_height))  # 30 FPS

        # Create a spinner generator
        spin = spinner()

        # Show the video processing status
        with st.empty():
            frame_count = 0
            start_time = time.time()

            while True:
                # Read a frame from the video
                ret, frame = cap.read()

                # If the frame was read successfully, ret will be True
                if not ret:
                    st.warning("End of video or error reading frame.")
                    break

                # Increment the frame counter
                frame_count += 1

                # Convert frame to JPEG (binary data) format
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                # Send the frame (as an image) to the prediction API
                response = requests.post(prediction_endpoint, headers=headers, data=img_bytes)

                # Show a spinner in the terminal while waiting for the API response
                st.text(f'Processing frame {frame_count}/{total_frames}... {next(spin)}')

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
                    st.error(f"Request failed for frame {frame_count} with status code {response.status_code}")
                    st.error(response.text)

                # Write the modified frame to the output video
                out.write(frame)

            # Release the video capture and writer objects
            cap.release()
            out.release()

            # Calculate total processing time
            end_time = time.time()
            total_time = end_time - start_time
            st.success(f"\nProcessing complete. Total time taken: {total_time:.2f} seconds.")

            # Display the processed video
            st.video(temp_output_path)

            # Cleanup temporary files
            shutil.rmtree(temp_input_path, ignore_errors=True)
            shutil.rmtree(temp_output_path, ignore_errors=True)
# Handle live camera feed or Live streaming endpoint
elif st.session_state.live_button:
    st.subheader("Live Camera Feed Processing or Live streaming endpoint")
    # Toggle between live camera feed and live stream URL input
    live_mode = st.radio("Select Live Mode", ["Live Stream URL", "Live Camera Feed"])
    # Handle live camera feed or live streaming endpoint
    if live_mode == "Live Stream URL":
        st.subheader("Live Stream URL Processing")

        # Input box for live stream URL (e.g., RTSP/HTTP stream)
        live_stream_url = st.text_input("Enter Live Stream URL (e.g., RTSP, HTTP)")

        if live_stream_url:
            # Open the live stream using OpenCV
            cap = cv2.VideoCapture(live_stream_url)

            if not cap.isOpened():
                st.error("Error: Could not access the live stream.")
                st.stop()

            # Create a spinner generator
            # spin = spinner()

            # Streamlit placeholders for video display
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Live Stream Feed (Raw)")
                live_stream_placeholder = st.empty()  # Placeholder for live stream feed

            with col2:
                st.subheader("Processed Feed")
                processed_stream_placeholder = st.empty()  # Placeholder for processed video feed

            # Start processing frames from the live stream
            while True:
                ret, frame = cap.read()

                # If frame is read successfully, ret will be True
                if not ret:
                    st.warning("Error reading from live stream.")
                    break

                # Update the live stream feed placeholder with the live video (raw)
                live_stream_placeholder.image(frame, channels="BGR", caption="Live Stream Feed", use_container_width=True)

                # Convert frame to JPEG (binary data) format
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                # Send the frame (as an image) to the prediction API
                response = requests.post(prediction_endpoint, headers=headers, data=img_bytes)

                # Show a spinner in the terminal while waiting for the API response
                # st.text(f'Processing frame... {next(spin)}')

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
                    st.error(f"Request failed with status code {response.status_code}")
                    st.error(response.text)

                # Update the processed stream feed placeholder with the frame that includes the prediction label
                processed_stream_placeholder.image(frame, channels="BGR", caption="Processed Frame", use_container_width=True)

            # Release the video capture object when done
            cap.release()

    elif live_mode == "Live Camera Feed":
        st.subheader("Live Camera Feed Processing")

        # Start button for live camera feed
        live_feed_button = st.button("Start Live Camera Feed")
        
        if live_feed_button:
            # Open webcam feed (index 0 for default webcam)
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Error: Could not access the camera.")
                st.stop()

            # Create a spinner generator
            # spin = spinner()

            # Streamlit placeholders for video display
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Camera Feed (Raw)")
                camera_placeholder = st.empty()  # Placeholder for live camera feed

            with col2:
                st.subheader("Processed Feed")
                processed_placeholder = st.empty()  # Placeholder for processed video feed

            # Start processing frames from the webcam
            while True:
                ret, frame = cap.read()

                # If frame is read successfully, ret will be True
                if not ret:
                    st.warning("Error reading from camera feed.")
                    break

                # Update the camera feed placeholder with the live video (raw)
                camera_placeholder.image(frame, channels="BGR", caption="Live Camera Feed", use_container_width=True)

                # Convert frame to JPEG (binary data) format
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                # Send the frame (as an image) to the prediction API
                response = requests.post(prediction_endpoint, headers=headers, data=img_bytes)

                # Show a spinner in the terminal while waiting for the API response
                # st.text(f'Processing frame... {next(spin)}')

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
                    st.error(f"Request failed with status code {response.status_code}")
                    st.error(response.text)

                # Update the processed frame placeholder with the frame that includes the prediction label
                processed_placeholder.image(frame, channels="BGR", caption="Processed Frame", use_container_width=True)

            # Release the video capture object when done
            cap.release()   


else:
     st.session_state.upload_button = st.button("Upload Video")
     st.session_state.live_button = st.button("Live Stream")
     st.rerun()