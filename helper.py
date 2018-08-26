import os
import shutil

import imutils as imutils
import scipy.misc

import helper_batch
import json
import urllib.request
import cv2
import numpy as np
import shutil
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
#
# def prepare_data(dir, data=[]):
#     file_names = os.listdir(dir)
#     for file in file_names:
#
#     return data

def prepare_data(dir, data):
    file_names = os.listdir(dir)
    for file in file_names:
        img = cv2.imread(dir + '/' + file, 0)
        if img is not None:
            res = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            data.append(res)
    return data

def retrieve_file(single_data):
    dir = 'training_set_500/' + single_data['External ID']
    os.mkdir(dir)
    f = open(dir + '/origin.jpg', 'wb')
    f.write(urllib.request.urlopen(single_data['Labeled Data']).read())
    f.close()
    for mask in single_data['Masks']:
        # f = open(dir + '/' + mask + '.jpg', 'wb')
        resp = urllib.request.urlopen(single_data['Masks'][mask])
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        rotated = imutils.rotate_bound(image, 90)
        cv2.imwrite(dir + '/' + mask + '.jpg', rotated)
        # f.write(image)
        # f.close()


def retriev_classify_set(single_data, dir):
    dir = os.path.join(dir, single_data['Label']['Recognize'])
    print(dir)
    if not os.path.exists(os.path.join(dir, single_data['External ID'])):
        f = open(os.path.join(dir, single_data['External ID']), 'wb')
        f.write(urllib.request.urlopen(single_data['Labeled Data']).read())
        f.close()


if __name__ == '__main__':
    # gen = helper_batch.gen_batch_function('data_road/training', (160, 576))
    dir = 'sharpness_set'
    with open('sharpness_set.json') as json_data:
        try:
            os.mkdir(os.path.join(dir, 'True'))
            os.mkdir(os.path.join(dir, 'False'))
        except FileExistsError as e:
            print("Directory Exist")
            pass

        d = json.load(json_data)
        for data in d:
            try:
                retriev_classify_set(data, dir)
            except Exception as e:
                print(e)
                pass



    # image = imutils.rotate_bound(scipy.misc.imread('training_set_2/IMG_1069.JPG/origin.jpg'), 90)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # for i, (x, y) in enumerate(gen(1)):
    #     if i == 1:
    #         print ("a")

