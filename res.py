from PIL import Image, ImageDraw, ImageFont
import os

font_path = 'KarnakPro-Bold.ttf'
output_folder = './res'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font_size = 20
font = ImageFont.truetype(font_path, font_size)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

background_color = (255, 255, 255)
text_color = (0, 0, 0)

for letter in letters:
    text_width, text_height = font.getbbox(letter)[2:4]

    # 创建灰度图像
    image = Image.new('L', (text_width + 8, text_height + 8), background_color[0])
    draw = ImageDraw.Draw(image)

    x = (image.width - text_width) // 2
    y = (image.height - text_height) // 2

    draw.text((x, y), letter, font=font, fill=text_color[0])

    # 对图像进行二值化处理
    threshold = 128  # 设置阈值
    image = image.point(lambda p: p > threshold and 255 or 0, '1')  # 应用阈值，转换为二值图像

    inverted_image = image.point(lambda p: 255 - p)

    prefix = 'up_' if letter.isupper() else 'down_'
    filename = f'{prefix}{letter}.png'

    inverted_image.save(os.path.join(output_folder, filename))
    # image.save(os.path.join(output_folder, filename))
    print(f'Saved {filename}')

print('Done!')