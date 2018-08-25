import os
import json
import numpy as np
import pandas as pd

annot_files = os.listdir('annot')
data = []

for file in annot_files:
    tmp = open('annot/' + file, 'r')
    tmp = json.load(tmp)
    data.extend(tmp)

result = {
    'full_path': [],
    'left': [],
    'top': [],
    'right': [],
    'bottom': [],
    'class': []
}
for datum in data:
    for annot in datum['annotations']:
        result['full_path'].append('images/' + datum['filename'])
        result['class'].append(annot['class'])
        left = int(annot['x'])
        top = int(annot['y'])
        right = left + int(annot['width'])
        bottom = top + int(annot['height'])
        result['left'].append(left)
        result['top'].append(top)
        result['right'].append(right)
        result['bottom'].append(bottom)

res = pd.DataFrame.from_dict(result)
cls_idx = {}
for file in annot_files:
    cls_idx[file] = res[res['class'] == file.split('.')[0]].index
    cls_idx[file] = np.array(cls_idx[file])
    np.random.shuffle(cls_idx[file])
    print(cls_idx[file][:5])

idx = annot_files[0]
det_train = res.iloc[cls_idx[idx][:40]]
det_test = res.iloc[cls_idx[idx][40:50]]
cls_data = res.iloc[cls_idx[idx][50:]]
for ci in annot_files[1:]:
    det_train = pd.concat([det_train, res.iloc[cls_idx[ci][:40]]], axis=0)
    det_test = pd.concat([det_test, res.iloc[cls_idx[ci][40:50]]], axis=0)
    cls_data = pd.concat([cls_data, res.iloc[cls_idx[ci][50:]]], axis=0)

det_train.to_csv('det_train.csv', index=False)
det_test.to_csv('det_test.csv', index=False)
cls_data.to_csv('cls_data.csv', index=False)
print(det_train.head())
