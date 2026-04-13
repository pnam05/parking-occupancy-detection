import cv2
import json
import numpy as np

rois = []
current_polygon = []  
mouse_x, mouse_y = 0, 0 

scale = 1.0       
pan_x = 0        
pan_y = 0         
is_panning = False
pan_ix, pan_iy = -1, -1

WIN_W, WIN_H = 1280, 720
IMG_W, IMG_H = 0, 0 

def get_real_coords(screen_x, screen_y):
    real_x = int((screen_x / WIN_W) * (IMG_W / scale) + pan_x)
    real_y = int((screen_y / WIN_H) * (IMG_H / scale) + pan_y)
    return real_x, real_y

def mouse_callback(event, x, y, flags, param):
    global current_polygon, rois, mouse_x, mouse_y
    global scale, pan_x, pan_y, is_panning, pan_ix, pan_iy

    real_x, real_y = get_real_coords(x, y)
    mouse_x, mouse_y = real_x, real_y

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: 
            scale *= 1.2
        else:         
            scale /= 1.2
        
        scale = max(1.0, min(scale, 10.0))
        
        pan_x = max(0, min(pan_x, IMG_W - int(IMG_W / scale)))
        pan_y = max(0, min(pan_y, IMG_H - int(IMG_H / scale)))

    elif event == cv2.EVENT_RBUTTONDOWN:
        is_panning = True
        pan_ix, pan_iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE and is_panning:
        dx = x - pan_ix
        dy = y - pan_iy
        pan_x -= int((dx / WIN_W) * (IMG_W / scale))
        pan_y -= int((dy / WIN_H) * (IMG_H / scale))
        
        pan_x = max(0, min(pan_x, IMG_W - int(IMG_W / scale)))
        pan_y = max(0, min(pan_y, IMG_H - int(IMG_H / scale)))
        pan_ix, pan_iy = x, y

    elif event == cv2.EVENT_RBUTTONUP:
        is_panning = False

    elif event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([real_x, real_y])
        
        if len(current_polygon) == 4:
            rois.append(list(current_polygon))
            current_polygon = []

video_path = './data/14191689_1920_1080_30fps.mp4' 
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, original_image = cap.read()
cap.release() 

if not ret or original_image is None:
    print("Lỗi: Không thể lấy frame từ video.")
    exit()

IMG_H, IMG_W = original_image.shape[:2]

window_name = 'ROI'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, WIN_W, WIN_H)
cv2.setMouseCallback(window_name, mouse_callback)

print("--- HƯỚNG DẪN SỬ DỤNG VẼ 4 ĐIỂM ---")
print("- CHUỘT TRÁI (Click): Click 4 góc của slot đỗ xe. Đủ 4 điểm sẽ tự lưu thành 1 slot.")
print("- CHUỘT PHẢI (Kéo thả): Di chuyển góc nhìn (Pan).")
print("- LĂN CHUỘT (hoặc phím +/-): Phóng to / Thu nhỏ (Zoom).")
print("- Phím 'z': Xóa điểm vừa click (nếu đang vẽ dở) hoặc xóa Slot vừa vẽ.")
print("- Phím 's': Lưu dữ liệu vào rois.json.")
print("- Phím 'q': Thoát.")

while True:
    canvas = original_image.copy()

    for idx, poly in enumerate(rois):
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        cv2.putText(canvas, str(idx+1), (poly[0][0], poly[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(current_polygon) > 0:
        for i in range(len(current_polygon)):
            cv2.circle(canvas, tuple(current_polygon[i]), 4, (0, 255, 255), -1)
            if i > 0:
                cv2.line(canvas, tuple(current_polygon[i-1]), tuple(current_polygon[i]), (0, 255, 255), 2)
        
        cv2.line(canvas, tuple(current_polygon[-1]), (mouse_x, mouse_y), (0, 165, 255), 1)

    view_w = int(IMG_W / scale)
    view_h = int(IMG_H / scale)
    
    x1 = int(pan_x)
    y1 = int(pan_y)
    x2 = min(IMG_W, x1 + view_w)
    y2 = min(IMG_H, y1 + view_h)
    
    cropped_view = canvas[y1:y2, x1:x2]

    display_img = cv2.resize(cropped_view, (WIN_W, WIN_H))

    cv2.putText(display_img, f"Zoom: {scale:.1f}x", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(display_img, f"Slots: {len(rois)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if len(current_polygon) > 0:
        cv2.putText(display_img, f"Dang ve diem: {len(current_polygon) + 1}/4", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow(window_name, display_img)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('z'):
        if len(current_polygon) > 0:
            current_polygon.pop()
            print(f"Đã xóa điểm cuối. Đang vẽ điểm thứ {len(current_polygon)+1}/4")
        elif len(rois) > 0:
            rois.pop()
            print(f"Undo! Còn lại {len(rois)} slots.")
            
    elif key == ord('s'):
        with open('rois.json', 'w') as f:
            json.dump(rois, f, indent=4)
        print(f"Đã lưu {len(rois)} slots vào rois.json!")
        
    elif key == ord('=') or key == ord('+'):  
        scale = min(10.0, scale * 1.2)
        pan_x = max(0, min(pan_x, IMG_W - int(IMG_W / scale)))
        pan_y = max(0, min(pan_y, IMG_H - int(IMG_H / scale)))
    elif key == ord('-'):                     
        scale = max(1.0, scale / 1.2)
        pan_x = max(0, min(pan_x, IMG_W - int(IMG_W / scale)))
        pan_y = max(0, min(pan_y, IMG_H - int(IMG_H / scale)))
    elif key == ord('q'):
        break

cv2.destroyAllWindows()