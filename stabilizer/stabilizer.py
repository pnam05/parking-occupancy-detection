import cv2
import numpy as np
import config

class VideoStabilizer:
    def __init__(self, first_frame):
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **config.FEATURE_PARAMS)
        self.p_initial = self.p0.copy() if self.p0 is not None else None
        
        self.smoothed_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def update(self, current_gray_frame):
        """Tính toán và trả về ma trận dịch chuyển cho frame hiện tại"""
        if self.p0 is not None and len(self.p0) > 10:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, current_gray_frame, self.p0, None, **config.LK_PARAMS
            )
            
            good_new = p1[st == 1]
            good_initial = self.p_initial[st == 1]

            if len(good_new) >= 4:
                M, inliers = cv2.estimateAffinePartial2D(
                    good_initial, good_new, cv2.RANSAC,
                    ransacReprojThreshold=3.0, 
                    maxIters=100
                )
                if M is not None:
                    self.smoothed_M = config.SMOOTHING_ALPHA * M + (1.0 - config.SMOOTHING_ALPHA) * self.smoothed_M

            self.old_gray = current_gray_frame.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            self.p_initial = good_initial.reshape(-1, 1, 2)

            if len(self.p0) < 100: 
                new_features = cv2.goodFeaturesToTrack(current_gray_frame, mask=None, **config.FEATURE_PARAMS)
                if new_features is not None:
                    M_inv = cv2.invertAffineTransform(self.smoothed_M)
                    new_initials = cv2.transform(new_features, M_inv)
                    self.p0 = np.vstack((self.p0, new_features))
                    self.p_initial = np.vstack((self.p_initial, new_initials))
                    
        return self.smoothed_M