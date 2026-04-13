import numpy as np
import cv2

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