import cv2
import torch
import numpy as np
from PIL import Image
import config
from model import load_parking_model, get_transforms
from stabilizer import VideoStabilizer
from utils import load_rois, draw_hud

def main():
    model = load_parking_model()
    infer_transforms = get_transforms()

    rois_data, total_slots = load_rois(config.ROI_PATH)
    
    raw_preds = [0] * total_slots
    confirmed_preds = [0] * total_slots
    consecutive_counts = [config.DEBOUNCE_THRESHOLD] * total_slots
    current_slot_idx = 0 

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret:
        print("[LỖI] Không thể đọc video!")
        return

    stabilizer = VideoStabilizer(first_frame)

    cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Parking System", 1280, 720)

    init_tensors = []
    for data in rois_data:
        pts_int = np.int32(data['pts_ref'])
        x, y, w, h = cv2.boundingRect(pts_int)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(first_frame.shape[1], x+w), min(first_frame.shape[0], y+h)
        
        local_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.fillPoly(local_mask, [pts_int - [x1, y1]], 255)
        
        crop = first_frame[y1:y2, x1:x2]
        masked_crop = cv2.bitwise_and(crop, crop, mask=local_mask)
        crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
        init_tensors.append(infer_transforms(Image.fromarray(crop_rgb)))

    if init_tensors:
        batch_tensor = torch.stack(init_tensors).to(config.DEVICE)
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
        for i, p in enumerate(preds.cpu().numpy()):
            raw_preds[i] = confirmed_preds[i] = p
            consecutive_counts[i] = config.DEBOUNCE_THRESHOLD 

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        H, W = frame.shape[:2]
        
        if frame_count % 2 != 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smoothed_M = stabilizer.update(gray)

            crop_tensors = []
            valid_indices = []

            for i in range(config.SLOTS_PER_FRAME):
                idx = (current_slot_idx + i) % total_slots
                pts_curr = cv2.transform(rois_data[idx]['pts_ref'], smoothed_M)
                pts_curr_int = np.int32(pts_curr)
                
                x, y, w, h = cv2.boundingRect(pts_curr_int)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x+w), min(H, y+h)
                w_c, h_c = x2 - x1, y2 - y1
                
                if w_c > 0 and h_c > 0:
                    local_mask = np.zeros((h_c, w_c), dtype=np.uint8)
                    cv2.fillPoly(local_mask, [pts_curr_int - [x1, y1]], 255)
                    crop = frame[y1:y2, x1:x2]
                    masked_crop = cv2.bitwise_and(crop, crop, mask=local_mask)
                    
                    crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
                    crop_tensors.append(infer_transforms(Image.fromarray(crop_rgb)))
                    valid_indices.append(idx)

            if crop_tensors:
                batch_tensor = torch.stack(crop_tensors).to(config.DEVICE)
                with torch.no_grad():
                    outputs = model(batch_tensor)
                    _, preds = torch.max(outputs, 1)
                
                for p_idx, current_pred in zip(valid_indices, preds.cpu().numpy()):
                    if current_pred == raw_preds[p_idx]:
                        consecutive_counts[p_idx] += 1
                    else:
                        consecutive_counts[p_idx] = 1
                        raw_preds[p_idx] = current_pred
                    
                    if consecutive_counts[p_idx] >= config.DEBOUNCE_THRESHOLD:
                        confirmed_preds[p_idx] = raw_preds[p_idx]

            current_slot_idx = (current_slot_idx + config.SLOTS_PER_FRAME) % total_slots

        draw_hud(frame, rois_data, confirmed_preds, stabilizer.smoothed_M)
        cv2.imshow("Smart Parking System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()