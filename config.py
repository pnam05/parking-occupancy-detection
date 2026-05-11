import torch
import cv2

VIDEO_PATH = "./data/14191689_1920_1080_30fps.mp4" 
ROI_PATH = "rois.json"
MODEL_WEIGHTS = "weights/best.pth"
OUTPUT_VIDEO_PATH = "output_parking.mp4"

CLASS_NAMES = ['empty', 'occupied']
SLOTS_PER_FRAME = 1
DEBOUNCE_THRESHOLD = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SMOOTHING_ALPHA = 0.15 
FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.03, minDistance=30, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))