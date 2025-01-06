import cv2
import os

def images_to_video(image_folder, output_video_path, frame_rate=30):
    # Get all the image file names in the folder (sorted by name)
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()

    # Check if there are any images in the folder
    if not images:
        raise ValueError(f"No images found in the folder: {image_folder}")

    # Read the first image to get the width and height
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4, use 'XVID' for .avi, etc.
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Iterate over all images and add them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video saved as: {output_video_path}")

# Example usage:
image_folder = '/Users/kavinkumarbaskar/project/threat-detection/threat-detection/test-images/roadaccidents'  # Folder containing image frames
output_video_path = 'output_video.mp4'  # Output video file path
frame_rate = 24  # Frame rate for the video

images_to_video(image_folder, output_video_path, frame_rate)
