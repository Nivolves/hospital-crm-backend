from PIL import Image


def binarization(img_path, output_path):
    image_f = Image.open(img_path)
    image = Image.new("RGB", image_f.size)
    image.paste(image_f)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    mutch_dict = {}
    for x in range(height):
        for y in range(width):
            p = pix[y, x]
            if p in mutch_dict.keys():
                mutch_dict.update({p: mutch_dict[p] + 1})
            else:
                mutch_dict.update({p: 0})
    max_l_pix = 0
    min_l_pix = 255
    full = height * width
    for x in range(height):
        for y in range(width):
            p = pix[y, x]
            if p[0] > max_l_pix and mutch_dict[p] > ((full / 100) * 1):
                max_l_pix = p[0]
            if p[0] < min_l_pix and mutch_dict[p] > ((full / 100) * 1):
                min_l_pix = p[0]
    max_val = 0
    max_key = None
    for key in mutch_dict.keys():
        val = mutch_dict[key]
        if max_val < val:
            max_val = val
            max_key = key

    for x in range(height):
        for y in range(width):
            p = pix[y, x]
            if p >= max_key:
                pix[y, x] = (0, 0, 0)
            else:
                pix[y, x] = (255, 255, 255)
    image.save(output_path)
