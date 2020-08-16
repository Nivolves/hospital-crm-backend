import cv2
import json
import os


def image_representation(img, top_pairs, brightness):
    top_colors = []
    for i in top_pairs:
        top_colors.append(i[0])
        top_colors.append(i[1])
    top_colors = list(set(top_colors))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] not in top_colors:
                img[i][j] = 0
            else:
                img[i][j] = img[i][j] << brightness
    return img


def get_transformed_image(diff_matrix, path_to_save, cur_dir, filename, brightness=2):
    with open(os.path.join(cur_dir, 'SystemBack/MaxFeatures/', filename)) as f:
        best_pairs = json.load(f)
    transformed_image = image_representation(diff_matrix, best_pairs['hor'], brightness)
    cv2.imwrite(path_to_save, transformed_image)
