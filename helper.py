import os
import imutils as imutils
import json
import urllib.request
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="choose for which model, data will be provided", action="store", dest="mode")
parser.add_argument("-f", "--file", help="file name of json with training data ", action="store", dest="file")

OUTPUT_DIRECTORY = "training_set_500"


def prepare_data(dir, data):
    file_names = os.listdir(dir)
    for file in file_names:
        img = cv2.imread(dir + '/' + file, 0)
        if img is not None:
            res = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            data.append(res)
    return data


def retrieve_file(single_data):
    dir = os.path.join(OUTPUT_DIRECTORY, single_data['External ID'])
    os.mkdir(dir)
    f = open(dir + '/origin.jpg', 'wb')
    f.write(urllib.request.urlopen(single_data['Labeled Data']).read())
    f.close()
    for mask in single_data['Masks']:
        resp = urllib.request.urlopen(single_data['Masks'][mask])
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        rotated = imutils.rotate_bound(image, 90)
        cv2.imwrite(dir + '/' + mask + '.jpg', rotated)


def retriev_classify_set(single_data, dir):
    dir = os.path.join(dir, single_data['Label']['Recognize'])
    print(dir)
    if not os.path.exists(os.path.join(dir, single_data['External ID'])):
        f = open(os.path.join(dir, single_data['External ID']), 'wb')
        f.write(urllib.request.urlopen(single_data['Labeled Data']).read())
        f.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.file is None:
        print("please provide file name with images names")
    else:
        filename = args.file
        if args.mode and args.mode == 'classifier':
            dir = 'sharpness_set'
            with open(filename) as json_data:
                try:
                    os.mkdir(dir)
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
        if args.mode and args.mode == 'segmentation':
            with open(filename) as json_data:
                try:
                    os.mkdir(OUTPUT_DIRECTORY)
                except:
                    pass
                d = json.load(json_data)
                for data in d:
                    try:
                        retrieve_file(data)
                    except:
                        pass
