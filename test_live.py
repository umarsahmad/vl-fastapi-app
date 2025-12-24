import cv2
import json
from datetime import datetime
from ultralytics import YOLO

# 1. Load your model
model = YOLO('models/best.pt')

# 2. Open Camera
cap = cv2.VideoCapture(0)

print("Starting Live Feed... Logs will appear below.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run Inference
    # We set 'verbose=False' to stop YOLO's default console spam 
    # so our custom logs are easy to read.
    results = model.predict(source=frame, conf=0.50, verbose=False)

    for result in results:
        # Get detected boxes
        boxes = result.boxes
        
        for box in boxes:
            # Extract data
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Create a conventional JSON record
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": "Defect Detected",
                "type": label,
                "confidence": round(confidence, 4)
            }
            
            # Print as JSON string
            print(json.dumps(log_entry))

        # 4. Display the visual feed
        annotated_frame = result.plot()
        cv2.imshow('Steel Quality Monitor', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()