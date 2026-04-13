import cv2
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import threading

# Config
VIDEO_PATH = "./data/14191689_1920_1080_30fps.mp4" 
ROI_PATH = "rois.json"
MODEL_WEIGHTS = "weights/best.pth"
REF_IMAGE_PATH = 'reference_frame.jpg'
CLASS_NAMES = ['empty', 'occupied']

AI_INTERVAL = 30     
ALIGN_INTERVAL = 90  
DEBOUNCE_THRESHOLD = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Đang khởi động trên thiết bị: {device}")

# Model
model = models.mobilenet_v3_small(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model = model.to(device)
model.eval() 

infer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Preprocess ROI
with open(ROI_PATH, "r") as f:
    rois = json.load(f)

rois_data = []
for roi in rois:
    pts = np.array(roi, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    
    local_pts = pts - np.array([x, y])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_pts], 255)
    
    M = cv2.moments(pts)
    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else tuple(pts[0])

    rois_data.append({
        'pts': pts.reshape((-1, 1, 2)),
        'bbox': (x, y, w, h),
        'mask': mask,
        'center': center
    })

# Image registration
ref_img = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if ref_img is None:
    print(f"[LỖI] Không tìm thấy ảnh gốc để neo camera: {REF_IMAGE_PATH}!")
    exit()

orb = cv2.ORB_create(2000)
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

M_lock = threading.Lock()
last_M = None
is_aligning = False

current_M = None        
SMOOTHING_ALPHA = 0.1

last_preds = [0] * len(rois)
is_inferencing = False

# Background workers
def update_alignment_matrix(current_frame):
    """Luồng 1: Tìm ma trận chống rung ngầm"""
    global last_M, is_aligning
    
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray_frame, None)
    
    if des_curr is None or len(des_curr) < 10:
        is_aligning = False
        return

    matches = matcher.match(des_curr, des_ref)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:max(4, int(len(matches) * 0.15))]

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            with M_lock:
                last_M = M
                
    is_aligning = False


raw_preds = [0] * len(rois)
confirmed_preds = [0] * len(rois)
consecutive_counts = [0] * len(rois)
def ai_worker(frame_copy):
    """Luồng 2: Chạy nhận diện AI ngầm"""
    global is_inferencing, raw_preds, confirmed_preds, consecutive_counts
    
    crop_tensors = []
    
    for data in rois_data:
        x, y, w, h = data['bbox']
        crop = frame_copy[y:y+h, x:x+w]
        
        masked_crop = cv2.bitwise_and(crop, crop, mask=data['mask'])
        
        crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
        tensor_img = infer_transforms(Image.fromarray(crop_rgb))
        crop_tensors.append(tensor_img)
        
    if len(crop_tensors) > 0:
        batch_tensor = torch.stack(crop_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
        
        preds_np = preds.cpu().numpy()
        
        for i, current_pred in enumerate(preds_np):
            if current_pred == raw_preds[i]:
                consecutive_counts[i] += 1
            else:
                consecutive_counts[i] = 1
                raw_preds[i] = current_pred
            
            if consecutive_counts[i] >= DEBOUNCE_THRESHOLD:
                confirmed_preds[i] = raw_preds[i]
                
    is_inferencing = False

# Main thread
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Parking System", 1280, 720)

ret, init_frame = cap.read()
if ret:
    crop_tensors = []
    for data in rois_data:
        x, y, w, h = data['bbox']
        crop = init_frame[y:y+h, x:x+w]
        masked_crop = cv2.bitwise_and(crop, crop, mask=data['mask'])
        crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
        tensor_img = infer_transforms(Image.fromarray(crop_rgb))
        crop_tensors.append(tensor_img)
    
    if len(crop_tensors) > 0:
        batch_tensor = torch.stack(crop_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
        
        init_preds = preds.cpu().numpy()

        for i, p in enumerate(init_preds):
            raw_preds[i] = p
            confirmed_preds[i] = p
            consecutive_counts[i] = DEBOUNCE_THRESHOLD 

frame_count = 0
print("[SYSTEM] Bắt đầu chạy hệ thống...")

while True:
    ret, raw_frame = cap.read()
    if not ret: 
        print("[SYSTEM] Đã hết video.")
        break

    if frame_count % ALIGN_INTERVAL == 0 and not is_aligning:
        is_aligning = True
        threading.Thread(target=update_alignment_matrix, args=(raw_frame.copy(),), daemon=True).start()

    with M_lock:
        target_M = last_M.copy() if last_M is not None else None
        
    if target_M is not None:
        if current_M is None:
            current_M = target_M 
        else:
            current_M = current_M * (1.0 - SMOOTHING_ALPHA) + target_M * SMOOTHING_ALPHA

        h, w = ref_img.shape
        frame = cv2.warpPerspective(raw_frame, current_M, (w, h))
    else:
        frame = raw_frame.copy()

    display = frame.copy()

    if frame_count % AI_INTERVAL == 0 and not is_inferencing:
        is_inferencing = True
        threading.Thread(target=ai_worker, args=(frame.copy(),), daemon=True).start()

    empty_count = 0
    occupied_count = 0

    for i, data in enumerate(rois_data):
        pred_label = confirmed_preds[i]

        if pred_label == 0: 
            color = (0, 255, 0)
            empty_count += 1
        else:               
            color = (0, 0, 255)
            occupied_count += 1

        cv2.polylines(display, [data['pts']], isClosed=True, color=color, thickness=2)
        cv2.putText(display, str(i+1), data['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.rectangle(display, (10, 10), (450, 130), (0, 0, 0), -1)
    cv2.putText(display, f"Total Slots: {len(rois)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"Empty (Green): {empty_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"Occupied (Red): {occupied_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    if is_inferencing:
        cv2.putText(display, "AI Computing...", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if is_aligning:
        cv2.putText(display, "Aligning Camera...", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.imshow("Smart Parking System", display)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Đã tắt an toàn.")