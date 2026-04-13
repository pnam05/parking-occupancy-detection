import cv2
import json
import numpy as np
import os
from util import crop_polygon, get_slot_center, point_in_polygon

VIDEO_PATH = "./data/14191689_1920_1080_30fps.mp4" 
ROI_PATH = "rois.json"
OUTPUT_DIR = "dataset/train"
IMG_SIZE = 64  

with open(ROI_PATH, "r") as f:
    rois = json.load(f)

os.makedirs(os.path.join(OUTPUT_DIR, "empty"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "occupied"), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

slot_labels = {} 
selected_slot = None


def mouse_callback(event, x, y, flags, param):
    global selected_slot
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, roi in enumerate(rois):
            if point_in_polygon((x, y), roi):
                selected_slot = i
                print(f"Selected slot {i} → nhấn E=empty, O=occupied")
                break

cv2.namedWindow("Data Annotation Tool", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Data Annotation Tool", 1280, 720)
cv2.setMouseCallback("Data Annotation Tool", mouse_callback)

COLOR = {
    "empty": (0, 255, 0),      
    "occupied": (0, 0, 255),  
    None: (200, 200, 200),    
}

print("=== HƯỚNG DẪN ===")
print("Click vào slot → nhấn E (empty) hoặc O (occupied)")
print("S = Save tất cả slot đã label & TỰ ĐỘNG NEXT FRAME")
print("A = Auto-save frame toàn EMPTY & TỰ ĐỘNG NEXT FRAME")
print("Space / N = Bỏ qua frame này, Next frame (+30 frames)")
print("Q = Quit")

ret, frame = cap.read()

while True:
    display = frame.copy()

    for i, roi in enumerate(rois):
        pts = np.array(roi, np.int32)
        label = slot_labels.get(i, None)
        color = COLOR[label]

        if i == selected_slot:
            color = (0, 255, 255)  

        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

        center = get_slot_center(roi)
        text = "E" if label == "empty" else ("O" if label == "occupied" else str(i))
        cv2.putText(display, text, center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)

    labeled = len(slot_labels)
    total = len(rois)
    cv2.putText(display, f"Frame {frame_count} | Labeled: {labeled}/{total} | Selected: {selected_slot}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Data Annotation Tool", display)
    
    key = cv2.waitKey(30) & 0xFF

    if key == ord('e') and selected_slot is not None:
        slot_labels[selected_slot] = "empty"

    elif key == ord('o') and selected_slot is not None:
        slot_labels[selected_slot] = "occupied"

    elif key == ord('s'):
        saved = 0
        for idx, label in slot_labels.items():
            crop = crop_polygon(frame, rois[idx])
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            filename = f"{label}_{frame_count}_{idx}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, label, filename), crop)
            saved += 1
        print(f"[Saved] {saved} crops từ frame {frame_count}")
        
        slot_labels.clear()
        selected_slot = None
        for _ in range(30): cap.read()
        ret, frame = cap.read()
        frame_count += 31
        if not ret: break

    elif key == ord('a'):
        for i, roi in enumerate(rois):
            crop = crop_polygon(frame, roi)
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            filename = f"empty_{frame_count}_{i}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "empty", filename), crop)
        print(f"[Auto EMPTY] Saved {len(rois)} crops từ frame {frame_count}")
        
        slot_labels.clear()
        selected_slot = None
        for _ in range(30): cap.read()
        ret, frame = cap.read()
        frame_count += 31
        if not ret: break

    elif key == ord(' ') or key == ord('n'):
        slot_labels.clear()
        selected_slot = None
        for _ in range(30): cap.read()
        ret, frame = cap.read()
        frame_count += 31
        if not ret: break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()