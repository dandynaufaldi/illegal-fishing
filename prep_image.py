import cv2
import os
import numpy as np
import pandas as pd


def crop(path: str, left: int, top: int, right: int, bottom: int):
    image = cv2.imread(path)


def process_image(db: str):
    data = pd.read_csv('det_{}.csv'.format(db))
    paths = data['full_path'].values


if __name__ == '__main__':
    process_image('train')
    process_image('test')
