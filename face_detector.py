from os import path
import cv2
import dlib

PREDICTOR_PATH = 'data/shape_predictor_68_face_landmarks.dat'
DETECTOR_PATH = 'data/mmod_human_face_detector.dat'

class FaceDetector:
    def detect(self):
        raise NotImplementedError


class DlibCVFaceDetector(FaceDetector):

    def __init__(self, detector_type='cnn'):
        self.detector_type = detector_type
        if detector_type == 'cnn':
            self.face_detector = dlib.cnn_face_detection_model_v1(DETECTOR_PATH)
        else:
            self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def detect(self, image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.face_detector(image, 1)
        if (results is None):
            return ()

        rects = []
        for face in results:
            if self.detector_type == 'cnn':
                left = face.rect.left()
                top = face.rect.top()
                right = face.rect.right()
                bottom = face.rect.bottom()
            else:
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()

            rects.append((top, bottom, left, right))

        return rects

    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
     
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
     
        # return the list of (x, y)-coordinates
        return coords

    def get_68_points(self, image, rectangle):
        #print(rectangle)
        shape = self.landmark_predictor(image, rectangle)
        shape = shape_to_np(shape)
        return shape


