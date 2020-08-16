import sys
import json
import os
from FindLiverClass import get_classification_results
from MakeRepresentation import get_transformed_image
from MakeBinarization import binarization

if __name__ == "__main__":
    # Входные аргументы
    link = sys.argv[1]
    task_type = sys.argv[2]
    sensor_type = sys.argv[3]
    path_to_save1 = sys.argv[4]
    path_to_save2 = sys.argv[5]

    cur_dir = "".join(
        os.path.abspath(__file__).rsplit(__file__)
    )  # путь текущей директории
    filename = sensor_type + ".json"  # название json файла

    # Основной код
    res, diff_matrix = get_classification_results(
        {
            "path": link,
            "task_type": task_type,
            "sensor_type": sensor_type,
            "cur_dir": cur_dir,
            "filename": filename,
        }
    )
    print(json.dumps(res))
    get_transformed_image(diff_matrix, path_to_save1, cur_dir, filename)
    binarization(link, path_to_save2)
