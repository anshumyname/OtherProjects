import copy
import cv2
import mediapipe as mp
import time
import imutils


class PoseDectector():
    def __init__(self, mode=False, model_complex=False, smoothness=True, enable_segmentation=False, smooth_segmentation=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.model_complex = model_complex
        self.smoothness = smoothness
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.model_complex, self.smoothness, self.enable_segmentation,
                                     self.smooth_segmentation, self.detection_confidence, self.tracking_confidence)
        self.POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            right_Person = copy.deepcopy(self.results.pose_landmarks)
            left_Person = copy.deepcopy(self.results.pose_landmarks)
            
            for i in range(len(self.results.pose_landmarks.landmark)):
                right_Person.landmark[i].x += (300/img.shape[1])
            
            img = self.draw_custom_landmarks(img, right_Person, color=(0,255,0))

            for i in range(len(self.results.pose_landmarks.landmark)):
                left_Person.landmark[i].x -= (300/img.shape[1])
            
            img = self.draw_custom_landmarks(img, left_Person, color=(0,0,255))

            # if draw:
                
                # self.draw_custom_landmarks(img, right_Person, color=(0,0,255))

        return img

    def draw_custom_landmarks(self, img, landmarks, color,connections = True):
        idx_to_coordinates = {}
        image_rows, image_cols, _ = img.shape
        for idx, landmark in enumerate(landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < 0.5) or (landmark.HasField('presence') and landmark.presence < 0.5)):
                continue

            landmark_px = (int(landmark.x*image_cols),
                           int(landmark.y*image_rows))
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

        if connections:
            num_landmarks = len(landmarks.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in self.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    thickness= 10
                    cv2.line(img, idx_to_coordinates[start_idx],
                            idx_to_coordinates[end_idx], color,thickness)

        return img




def main():
    cap = cv2.VideoCapture(0)

    detector = PoseDectector()
    pTime = 0
    while True:
        success, img = cap.read()
        img = imutils.resize(img, width=1200)
        detector.findPose(img)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
