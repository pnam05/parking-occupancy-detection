import numpy as np
import cv2
import json

def crop_polygon(image, polygon):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    crop = masked[y:y+h, x:x+w]
    return crop

def get_slot_center(polygon):
    pts = np.array(polygon, np.int32)
    M = cv2.moments(pts)
    if M["m00"] == 0:
        return tuple(pts[0])
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def point_in_polygon(point, polygon):
    pts = np.array(polygon, np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0

def load_rois(filepath):
    with open(filepath, "r") as f:
        rois = json.load(f)
    
    rois_data = []
    for roi in rois:
        pts_ref = np.array(roi, dtype=np.float32).reshape((-1, 1, 2))
        rois_data.append({'pts_ref': pts_ref})
    
    return rois_data, len(rois)

def draw_hud(frame, rois_data, confirmed_preds, smoothed_M):
    empty_count = 0
    occupied_count = 0
    total_slots = len(rois_data)

    for i in range(total_slots):
        pred_label = confirmed_preds[i]
        pts_curr = cv2.transform(rois_data[i]['pts_ref'], smoothed_M)
        pts_curr_int = np.int32(pts_curr)
        
        M_mom = cv2.moments(pts_curr_int)
        cx = int(M_mom["m10"] / M_mom["m00"]) if M_mom["m00"] != 0 else pts_curr_int[0][0][0]
        cy = int(M_mom["m01"] / M_mom["m00"]) if M_mom["m00"] != 0 else pts_curr_int[0][0][1]

        if pred_label == 0: 
            color = (0, 255, 0)
            empty_count += 1
        else:               
            color = (0, 0, 255)
            occupied_count += 1

        cv2.polylines(frame, [pts_curr_int], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.rectangle(frame, (10, 10), (450, 130), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Slots: {total_slots}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Empty (Green): {empty_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Occupied (Red): {occupied_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)