import numpy as np
import argparse
import dlib
import cv2
import face_detector
import warping
import utils
import os
from tqdm import tqdm
import glob
from PIL import Image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")

def extractor(image_path, detector=None, predictor=None, save_dir=''):
    scale = 1
    img = Image.open(image_path).convert('RGB')
    g_img = np.array(img.convert('L'))
    img = np.array(img)
    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale

    dets = detector(img, 1)

    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = warping.extract_eye_center(shape, 'left')
        right_eye = warping.extract_eye_center(shape, 'right')

        M = warping.get_rotation_matrix(left_eye, right_eye)


        rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)
        rotated_g = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
        rotated_det = detector(rotated_g, 1)
        shape = predictor(rotated_g, rotated_det[0])
        
        cropped = warping.crop_image(Image.fromarray(rotated), shape)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        
        save_name = os.path.join(save_dir,
                                 '{}_{}'.format(i, os.path.basename(image_path)))
        
        result_img = utils.add_qrcode_in_image(cropped, save_name)

        result_img.save(save_name)

def main(args):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
    images = []
    if os.path.isdir(args['image']):
        images = glob.glob(os.path.join(args['image'], '*.*'))
    else:
        images.append(args['image'])

    for input_image in tqdm(images):
        save_dir = os.path.join('/'.join(input_image.split('/')[:-1]),
                                'warped')
        extractor(input_image, detector, predictor, save_dir)

if __name__ == '__main__':
    args = vars(ap.parse_args()) 
    print(args)
    main(args)