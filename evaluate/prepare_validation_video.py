# -*- coding: utf-8 -*-
# FIRST TERMINAL
# cd CARLA_0.9.9.3/
# . ./.venv/bin/activate
# ./CarlaUE4.sh -windowed  -ResX=1080 -ResY =1920 -carla -server  -fps=100-quality -level=high
# SECOND TERMINAL
# cd CARLA_0.9.9.3/
# . ./.venv/bin/activate
# export CARLA_ROOT=~/CARLA_0.9.9.3/
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
# cd PythonAPI/examples/
# python3 manual_control.py --res 608x608
# python3 prepare_validation_video.py 

import cv2
import argparse
#from utils.utils import *
from os import listdir
from os.path import isfile, join
from time import sleep

def main():
    #img = [f for f in listdir(path) if isfile(join(path, f))]
    dirf = listdir(path)
    dirf.sort()
    sorted(dirf)
    print(dirf)
    img = []
    for files in dirf:
        if('.png' in files):
            img.append(files) 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args["out"] + args["in"], fourcc , 30.0, (1600,  900))
    
    print(path + str(img[0]))
    print(len(img))
    sleep(1)

    for i in range(0, len(img)):
        img_name = join(path, img[i])
        frame = cv2.imread(img_name)
        frame = cv2.resize(frame, (1600, 900), interpolation = cv2.INTER_CUBIC) 
        cv2.imshow('frame', frame)
        i+=1
        print(i)
        out.write(frame)
        cv2.imwrite("videos/" + test + "/" + str(i) + ".png", frame)
        if cv2.waitKey(30) == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, default='carla_dataset/t4/test/t4_5.mp4', help='File path')
    parser.add_argument('--out', type=str, default='videos/', help='Output folder')
    opt = parser.parse_args()
    main()