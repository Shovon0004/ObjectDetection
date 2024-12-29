import cv2
import numpy as np
import time
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' (nano version) or other versions like 'yolov8s.pt'
        self.confidence_threshold = confidence_threshold

        # Generate random colors for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(80, 3)).tolist()

    def process_frame(self, frame):
        # Convert frame to RGB (YOLOv8 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = self.model(frame_rgb)

        # Get detections from results
        detections = results[0].boxes.data.cpu().numpy()  # Extract detection data

        # Draw detections
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            if confidence >= self.confidence_threshold:
                # Get coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class information
                class_id = int(class_id)
                class_name = self.model.names[class_id]

                # Get color for this class
                color = self.colors[class_id % len(self.colors)]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Create label
                label = f'{class_name}: {confidence:.2f}'

                # Get label size
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                # Draw label background
                cv2.rectangle(annotated_frame,
                              (x1, y1 - label_height - 10),
                              (x1 + label_width + 10, y1),
                              color, -1)

                # Draw label text
                cv2.putText(annotated_frame, label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

        # Add FPS counter
        fps = int(1 / (time.time() - self.last_time))
        self.last_time = time.time()
        cv2.putText(annotated_frame, f'FPS: {fps}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        return annotated_frame

    def run_detection(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Set higher resolution if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.last_time = time.time()

        print("Starting object detection... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            annotated_frame = self.process_frame(frame)

            # Display result
            cv2.imshow('Object Detection', annotated_frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Create detector with 0.5 confidence threshold
        detector = ObjectDetector(confidence_threshold=0.5)

        # Run detection
        detector.run_detection()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
