import cv2
import torch
import numpy as np
from PIL import Image
import time

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.confidence_threshold = confidence_threshold
        
        # Generate random colors for different classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(80, 3)).tolist()

    def process_frame(self, frame):
        # Convert frame to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Perform detection
        results = self.model(pil_image)
        
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Draw detections
        annotated_frame = frame.copy()
        for idx, detection in detections.iterrows():
            if detection['confidence'] >= self.confidence_threshold:
                # Get coordinates
                x1, y1 = int(detection['xmin']), int(detection['ymin'])
                x2, y2 = int(detection['xmax']), int(detection['ymax'])
                
                # Get class information
                class_name = detection['name']
                confidence = detection['confidence']
                
                # Get color for this class
                color = self.colors[idx % len(self.colors)]
                
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