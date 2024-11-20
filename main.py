"""
 @Author: Envision
 @Github: ElapsingDreams
 @Gitee: ElapsingDreams
 @Email: None
 @FileName: main.py
 @DateTime: 2024/11/19 18:24
 @SoftWare: PyCharm
"""

import cv2
import numpy as np
import os


def load_image(image_path):
    img = cv2.imread(image_path)
    return img


def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def binarize(image, threshold=127):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def build_gaussian_pyramid(image, levels=6):
    # 高斯金字塔
    pyramid = [image]
    for i in range(1, levels):
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


def find_best_match(pyramid, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
    # 金字塔找到最佳匹配
    best_match = None
    for level in pyramid:
        match = template_matching(level, template, method, threshold)
        if match is not None:
            max_val, max_loc, (th, tw) = match
            if best_match is None or max_val > best_match[0]:
                best_match = (max_val, max_loc, (th, tw), level.shape)
    return best_match


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


def main(image_path, template_paths):
    image = load_image(image_path)
    gray = to_gray(image)
    binary_image = binarize(gray)
    cv2.imwrite('binary_image.png', binary_image)
    templates = [to_gray(load_image("./res/" + path)) for path in template_paths]
    templates = [binarize(template) for template in templates]

    pyramid = build_gaussian_pyramid(binary_image)

    matches = []
    for template in templates:
        match = find_best_match(pyramid, template)
        matches.append(match)
        print(match)

    result_image = draw_matches(image, matches)

    cv2.imwrite('MatchedImage.png', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    template_paths = os.listdir("./res")
    main('text_image.png', template_paths)