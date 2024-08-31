import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict, deque
import torch

if torch.cuda.is_available() is True:
    model = YOLO("yolov8s.pt").to("cuda")

    # Define ROI regions
    area_up = np.array([[658, 418], [783, 401], [1040, 533], [861, 552]], dtype=np.int32)
    area_down = np.array([[727, 965], [1132, 890], [1339, 1072], [799, 1074]], dtype=np.int32)
    area_left = np.array([[5, 695], [330, 660], [350, 690], [5, 740]], dtype=np.int32)
    area_right = np.array([[1560, 640], [1809, 620], [1900, 700], [1650, 739]], dtype=np.int32)

    up = {}
    down = {}
    left = {}
    right = {}

    up_counter = 0
    down_counter = 0
    left_counter = 0
    right_counter = 0

    cap = cv.VideoCapture("intersection.mp4")
    model = YOLO("yolov8s.pt")

    # Get the frame width and height
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    output_filename = "output_intersection.mp4"
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_filename, fourcc, 30, (frame_width, frame_height))

    # Define vehicle types and tracking history
    vehicles = {1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}
    track_hist = defaultdict(lambda: deque(maxlen=25))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, conf=0.1)[0]
        boxes = np.array(results.boxes.data.tolist(), dtype="int")

        for box in boxes:
            x1, y1, x2, y2, track_id, score, class_id = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            up_result = cv.pointPolygonTest(area_up, (center_x, center_y), measureDist=False)
            down_result = cv.pointPolygonTest(area_down, (center_x, center_y), measureDist=False)
            left_result = cv.pointPolygonTest(area_left, (center_x, center_y), measureDist=False)
            right_result = cv.pointPolygonTest(area_right, (center_x, center_y), measureDist=False)

            # Check if a vehicle in ROI
            if up_result == 1 and track_id not in up:
                up[track_id] = (x1, y1, x2, y2)
                up_counter += 1

            if down_result == 1 and track_id not in down:
                down[track_id] = (x1, y1, x2, y2)
                down_counter += 1

            if left_result == 1 and track_id not in left:
                left[track_id] = (x1, y1, x2, y2)
                left_counter += 1

            if right_result == 1 and track_id not in right:
                right[track_id] = (x1, y1, x2, y2)
                right_counter += 1

            if class_id in vehicles:
                track = track_hist[track_id]
                track.append((center_x, center_y))

                points = np.hstack(track).astype("int32").reshape((-1, 1, 2))
                cv.polylines(frame, [points], isClosed=False, color=(255, 255, 255), thickness=7)
                cv.polylines(frame, [points], isClosed=False, color=(0, 0, 0), thickness=5)
                cv.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv.rectangle(frame, (x1 - 1, y1 - 35), (x2, y1), (255, 0, 0), -1)
                cv.putText(frame, f"#{vehicles[class_id]}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.putText(frame, f"#{track_id}",
                           (x2 - cv.getTextSize(f"#{track_id}", cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0], y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.rectangle(frame, (1600, 0), (1920, 200), (235, 206, 135), -1)

        # Counters
        cv.putText(frame, f"Up Counter: {up_counter}", (1610, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(frame, f"Right Counter: {right_counter}", (1610, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(frame, f"Down Counter: {down_counter}", (1610, 130), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(frame, f"Left Counter: {left_counter}", (1610, 170), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw ROIs
        cv.polylines(frame, [area_up], isClosed=True, color=(0, 255, 0), thickness=2)
        cv.polylines(frame, [area_down], isClosed=True, color=(0, 0, 255), thickness=2)
        cv.polylines(frame, [area_left], isClosed=True, color=(255, 0, 0), thickness=2)
        cv.polylines(frame, [area_right], isClosed=True, color=(0, 255, 255), thickness=2)

        # Write the frame to the video file
        out.write(frame)

        # Display the frame
        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

    # Print the counters
    print(f"Up Counter: {up_counter}")
    print(f"Down Counter: {down_counter}")
    print(f"Left Counter: {left_counter}")
    print(f"Right Counter: {right_counter}")

    # Visualize results with matplotlib
    directions = ["Up", "Down", "Left", "Right"]
    counts = [up_counter, down_counter, left_counter, right_counter]

    plt.figure(figsize=(8, 6))
    plt.bar(directions, counts, color=["green", "red", "blue", "yellow"])

    plt.title("Vehicle Count by Direction")
    plt.xlabel("Direction")
    plt.ylabel("Number of Vehicles")

    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)

    plt.show()

else:
    print("Please activate cuda")
