"""
 @Author: Envision
 @Github: ElapsingDreams
 @Gitee: ElapsingDreams
 @Email: None
 @FileName: main2.py
 @DateTime: 2024/11/19 18:55
 @SoftWare: PyCharm
"""
import os

import cv2
import numpy as np


def load_image(image_path):
    img = cv2.imread(image_path)
    return img


def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def build_gaussian_pyramid(image, levels=6):
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def template_matching(image, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return None

    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        return max_val, max_loc, template.shape
    return None


def find_best_match(pyramid, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.87):
    best_match = None
    best_scale = None
    for level, image in enumerate(pyramid):
        match = template_matching(image, template, method, threshold)
        if match is not None:
            max_val, max_loc, (th, tw) = match
            if best_match is None or max_val > best_match[0]:
                best_match = (max_val, max_loc, (th, tw), image.shape)
                best_scale = 1 / (2 ** level)
    return best_match, best_scale


def draw_matches(image, matches):
    for match in matches:
        if match is not None:
            max_val, max_loc, (th, tw), (level_h, level_w) = match
            scale_x = image.shape[1] / level_w
            scale_y = image.shape[0] / level_h
            top_left = (int(max_loc[0] * scale_x), int(max_loc[1] * scale_y))
            bottom_right = (int((max_loc[0] + tw) * scale_x), int((max_loc[1] + th) * scale_y))
            cv2.rectangle(image, top_left, bottom_right, 255, 2)
    return image


def main(image_path, template_paths, best_scale=None):
    image = load_image(image_path)
    gray = to_gray(image)
    templates = [to_gray(load_image("./res/" + path)) for path in template_paths]

    if best_scale is not None:
        scaled_template = cv2.resize(templates[0], None, fx=best_scale, fy=best_scale)
        match = template_matching(gray, scaled_template)
        if match is not None:
            print("Using recorded best scale:", best_scale)
            result_image = draw_matches(image, [match])
            cv2.imshow('Matched Image', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    pyramid = build_gaussian_pyramid(gray)
    matches = []
    scales = []

    for template in templates:
        match, scale = find_best_match(pyramid, template)
        matches.append(match)
        scales.append(scale)

    result_image = draw_matches(image, matches)
    cv2.imshow('Matched Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 记录最佳匹配尺度
    if scales[0] is not None:
        with open('best_scale.txt', 'w') as f:
            f.write(str(scales[0]))


if __name__ == '__main__':
    template_paths = os.listdir("./res")
    try:
        with open('best_scale.txt', 'r') as f:
            best_scale = float(f.read().strip())
    except FileNotFoundError:
        best_scale = None

    main('text_image.png', template_paths, best_scale)
