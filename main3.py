"""
 @Author: Envision
 @Github: ElapsingDreams
 @Gitee: ElapsingDreams
 @Email: None
 @FileName: main3.py
 @DateTime: 2024/11/21 19:30
 @SoftWare: PyCharm
"""
import cv2
import os, json
import numpy as np


def match_templates(image_path, template_dir):
    locs = []

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    template_files = [f for f in os.listdir(template_dir) if f.startswith('up_') and f.endswith('.png')]

    for template_file in template_files:
        template_path = os.path.join(template_dir, template_file)
        template = cv2.imread(template_path, 0)
        _, template_h = template.shape

        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.90

        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            print(pt)
            locs.append([pt, template_file[:-4], template_h])
            h, w = template.shape
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
            cv2.putText(image, template_file[:-4], (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # cv2.imshow('Template Matcher', image)
    cv2.imwrite('MatchedImage.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return locs


def split_line(locs, index_path):
    with open(index_path) as f:
        index = json.load(f)
    # [[(array([129, 229]), array([230, 130])), 'up_A', 20],  ...
    locs.sort(key=lambda x: (x[0][1], x[0][0]))
    # print(locs)
    for i, a in enumerate(locs):
        if i > 0:
            if (locs[i][0][1] - locs[i - 1][0][1]) > ((locs[i][2] + locs[i - 1][2]) / 2):
                print()

        print(index["data"][a[1]], end=" ")


split_line(
    match_templates('test2.jpeg', './res'),
    "./res/index.json"
)
