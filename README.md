# Intersection Vehicle Counter

https://github.com/user-attachments/assets/b120218a-1bf0-455e-9883-fcab824d54bd

This project is a vehicle tracking and counting system that utilizes YOLOv8 for real-time object detection and tracking across predefined regions of interest (ROI) in a video of a traffic intersection.

## Features

- **Real-time Vehicle Detection and Tracking:** Detects and tracks vehicles in a video stream using YOLOv8.
- **Region-based Counting:** Counts vehicles that pass through four predefined ROIs (up, down, left, and right).
- **Visual Output:** Draws bounding boxes around detected vehicles, tracks their paths, and visualizes the count results on the video frame.
- **Final Count Visualization:** Displays the total count of vehicles in each direction using a bar chart.

## Project Structure

- `main.py`: Main script for running the vehicle counting system.
- `intersection.mp4`: Input video file of a traffic intersection.
- `output_intersection.mp4`: Output video file with the counted vehicles and tracked paths.
- `requirements.txt`: Required Python packages.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YusufCelebii/Intersection-Anylsis-with-YOLOv8.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, ensure you have an appropriate GPU enabled and execute the script as follows:

## Notes
- Modify the ROI coordinates in the script if you are using a different video or need different regions of interest.
