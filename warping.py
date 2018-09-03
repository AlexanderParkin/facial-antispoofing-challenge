import numpy as np
import cv2
import os


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
NOSE_INDICES = [27,30]
CHIN_INDICES = [8]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_point(shape, indices):
    points = map(lambda i: shape.part(i), indices)
    return list(points)

def extract_center(shape, indices):
    points = extract_point(shape, indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // len(indices), sum(ys) // len(indices)

def extract_left_eye_center(shape):
    return extract_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, shape):
    nose_center = np.array(extract_center(shape, NOSE_INDICES))
    chin = np.array(extract_center(shape, CHIN_INDICES))
    half_height = int(numpy.linalg.norm(nose_center - chin))
    left = nose_center[0] - half_height
    top = nose_center[1] - half_height
    right = nose_center[0] + half_height
    bottom = nose_center[1] + half_height
    return image.crop((left, top, right,bottom))