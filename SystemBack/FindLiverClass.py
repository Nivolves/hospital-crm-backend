import os
import json
import numpy as np
import pandas as pd
import FeaturesStack as FS


def calculate_gmdh_model(img_f):
    if task_type == "1":
        if sensor_type == "convex":
            prob = (
                -0.946477
                + img_f["std_vert"] * np.cbrt(img_f["P95(1)_vert"]) * 0.0171222
                + np.power(img_f["balx2_hor"], 3)
                * np.sin(img_f["dif12_hor"])
                * (-1.583e-05)
                + img_f["P5_vert"] * np.cos(img_f["pair6664_vert"]) * (-0.007739)
                + np.cbrt(img_f["x2_vert"]) * np.cbrt(img_f["balx2_vert"]) * 0.0831053
                + np.cos(img_f["pair3947_hor"]) * np.cos(img_f["dif12_vert"]) * 0.413282
                + np.cos(img_f["pair4639_hor"])
                * np.cos(img_f["pair6967_vert"])
                * (-0.141326)
                + np.cbrt(img_f["maxfreq_hor"])
                * np.cbrt(img_f["mean(1)_vert"])
                * 0.396514
                + np.cos(img_f["pair4639_hor"])
                * np.arctan(img_f["pair5555_vert"])
                * 0.123721
                + np.sqrt(img_f["pair5045_hor"])
                * np.cos(img_f["pair4846_hor"])
                * (-0.110306)
                + np.sqrt(img_f["maxfreq_orig"])
                * np.power(img_f["balx2_hor"], 3)
                * 1.51139e-05
                + img_f["dif13_vert"] * np.cbrt(img_f["x2_orig"]) * 0.0276597
            )
        elif sensor_type == "linear":
            prob = (
                0.521463
                + np.cos(img_f["fractal_dim"])
                * np.arctan(img_f["pair1526_hor"])
                * (-0.510109)
                + np.cbrt(img_f["x2_orig"]) * np.arctan(img_f["std(3)_hor"]) * 0.320271
                + np.sin(img_f["Q1_vert"]) * np.cos(img_f["skew(2)_vert"]) * 0.347042
                + np.cbrt(img_f["median(2)_hor"]) * np.cos(img_f["Q3_vert"]) * 0.120014
                + np.sin(img_f["x1_orig"]) * np.sin(img_f["pair5050_vert"]) * 0.149371
                + np.power(img_f["kurt(1)_hor"], 2)
                * np.cos(img_f["pair2820_hor"])
                * 0.107874
                + np.power(img_f["pair4845_vert"], 3)
                * np.cos(img_f["mean(3)_vert"])
                * 1.95106e-05
                + np.cos(img_f["mean(3)_vert"])
                * np.arctan(img_f["mean(2)_hor"])
                * (-0.115669)
            )

        elif sensor_type == "reinforced_linear":
            prob = (
                0.564665
                + np.cbrt(img_f["pair2420_hor"])
                * np.arctan(img_f["P5(1)_hor"])
                * (-0.185308)
                + np.sin(img_f["std_hor"]) * np.sin(img_f["pair5359_vert"]) * 0.529036
                + np.cos(img_f["range_vert"])
                * np.cos(img_f["pair7878_vert"])
                * (-0.326662)
                + np.sin(img_f["pair6574_vert"])
                * np.cos(img_f["Q3(1)_hor"])
                * (-0.337944)
                + np.cos(img_f["IQR_vert"])
                * np.cos(img_f["median(2)_vert"])
                * (-0.237002)
                + np.sin(img_f["pair5359_vert"])
                * np.cos(img_f["median(2)_vert"])
                * (-0.118517)
                + np.cos(img_f["median(2)_vert"])
                * np.arctan(img_f["P5(1)_hor"])
                * 0.138423
                + np.cos(img_f["pair6574_vert"])
                * np.arctan(img_f["pair5649_vert"])
                * 0.051217
                + np.sin(img_f["pair5359_vert"])
                * np.arctan(img_f["x2_vert"])
                * 0.296591
                + img_f["dif23_vert"] * np.cos(img_f["dif23_vert"]) * 0.914249
            )
        else:
            prob = 0
    elif task_type == "2":
        prob = 0
    else:
        prob = 0
    return prob, 1 if prob < 0.5 else 2


def forest_prediction(img_f):
    if task_type == "1":
        with open(os.path.join(cur_dir, "SystemBack/SelfOrganizationForests/" + sensor_type + ".json")) as f:
            forest = json.load(f)
        ypl = []  # y_pred list
        for obj in forest:
            tree = pd.DataFrame(obj["tree"])
            leaf = 1
            index = 0
            flag = False
            y_pred = 0
            while not flag:
                node = tree.loc[index]
                if node["side"] == 1:
                    if img_f[node["feature"]] < node["threshold"]:
                        y_pred = 1
                    else:
                        y_pred = 2
                else:
                    if img_f[node["feature"]] < node["threshold"]:
                        y_pred = 2
                    else:
                        y_pred = 1
                try:
                    index = np.where(
                        (tree["previous_leaf"] == leaf)
                        & (tree["previous_direction"] == y_pred)
                    )[0][0]
                    leaf = tree.loc[index]["leaf_number"]
                except:
                    flag = True
            ypl.append(y_pred)
        ypl = np.asarray(ypl)
        ypl_sum = np.sum(ypl == 1) + np.sum(ypl == 2)
        if np.sum(ypl == 1) > np.sum(ypl == 2):
            y_pred = 1
            forest_prob = (np.sum(ypl == 1) / ypl_sum) * 100
        else:
            y_pred = 2
            forest_prob = (np.sum(ypl == 2) / ypl_sum) * 100
    elif task_type == "2":
        forest_prob = 0
        y_pred = 0
    else:
        forest_prob = 0
        y_pred = 0
    return forest_prob, y_pred


def get_mean_signs(img_f):
    if task_type == "1":
        if sensor_type == "convex":
            feature1, feature2, feature3 = (
                "cbrt(P95(1)_vert)",
                "cos(dif12_vert)",
                "std_vert",
            )
            threshold1, threshold2, threshold3 = (
                5.0132979349645845,
                0.6306169224667781,
                7.127663290343068,
            )
            value1, value2, value3 = (
                np.cbrt(img_f["P95(1)_vert"]),
                np.cos(img_f["dif12_vert"]),
                img_f["std_vert"],
            )
            if value1 < threshold1:
                res1 = "Печень в норме"
            else:
                res1 = "Печень не в норме"
            if value2 < threshold2:
                res2 = "Печень не в норме"
            else:
                res2 = "Печень в норме"
            if value3 < threshold3:
                res3 = "Печень в норме"
            else:
                res3 = "Печень не в норме"
        elif sensor_type == "linear":
            feature1, feature2, feature3 = (
                "cbrt(x2_orig)",
                "arctan(pair1526_hor)",
                "cos(fractal_dim)",
            )
            threshold1, threshold2, threshold3 = (
                0.6440777961495892,
                1.3522438545232742,
                0.41596845937104104,
            )
            value1, value2, value3 = (
                np.cbrt(img_f["x2_orig"]),
                np.arctan(img_f["pair1526_hor"]),
                np.cos(img_f["fractal_dim"]),
            )
            if value1 < threshold1:
                res1 = "Печень в норме"
            else:
                res1 = "Печень не в норме"
            if value2 < threshold2:
                res2 = "Печень не в норме"
            else:
                res2 = "Печень в норме"
            if value3 < threshold3:
                res3 = "Печень не в норме"
            else:
                res3 = "Печень в норме"
        elif sensor_type == "reinforced_linear":
            feature1, feature2, feature3 = (
                "cos(range_vert)",
                "cbrt(pair2420_hor)",
                "sin(pair5359_vert)",
            )
            threshold1, threshold2, threshold3 = (
                0.9998433086476912,
                1.6407957194770635,
                -0.5549728719823037,
            )
            value1, value2, value3 = (
                np.cos(img_f["range_vert"]),
                np.cbrt(img_f["pair2420_hor"]),
                np.sin(img_f["pair5359_vert"]),
            )
            if value1 < threshold1:
                res1 = "Печень в норме"
            else:
                res1 = "Печень не в норме"
            if value2 < threshold2:
                res2 = "Печень не в норме"
            else:
                res2 = "Печень в норме"
            if value3 < threshold3:
                res3 = "Печень не в норме"
            else:
                res3 = "Печень в норме"
        else:
            feature1, feature2, feature3 = "", "", ""
            threshold1, threshold2, threshold3 = 0, 0, 0
            value1, value2, value3 = 0, 0, 0
            res1, res2, res3 = 0, 0, 0
    elif task_type == "2":
        feature1, feature2, feature3 = "", "", ""
        threshold1, threshold2, threshold3 = 0, 0, 0
        value1, value2, value3 = 0, 0, 0
        res1, res2, res3 = 0, 0, 0
    else:
        feature1, feature2, feature3 = "", "", ""
        threshold1, threshold2, threshold3 = 0, 0, 0
        value1, value2, value3 = 0, 0, 0
        res1, res2, res3 = 0, 0, 0
    return [
        {"feature": feature1, "threshold": threshold1, "value": value1, "result": res1},
        {"feature": feature2, "threshold": threshold2, "value": value2, "result": res2},
        {"feature": feature3, "threshold": threshold3, "value": value3, "result": res3},
    ]


def get_all_features():
    with open(os.path.join(cur_dir, "SystemBack/Features/", filename)) as f:
        feature_names = json.load(f)["features"]
    with open(os.path.join(cur_dir, "SystemBack/BestGrad/", filename)) as f:
        best_grad = json.load(f)
    with open(os.path.join(cur_dir, "SystemBack/MaxFeatures/", filename)) as f:
        best_pairs = json.load(f)

    img_f = []

    # fractal dimension of image
    img_f.append(FS.mink_val(path))

    # initial matrix
    init_matrix = np.concatenate(FS.get_greyscale_matrix(path), axis=None)
    img_f.append((np.sum(init_matrix == np.amin(init_matrix)) / init_matrix.size) * 100)
    img_f.append((np.sum(init_matrix == np.amax(init_matrix)) / init_matrix.size) * 100)

    # glcm
    glcm = FS.get_glcm(init_matrix)
    img_f = FS.get_x1x2x3(
        glcm, img_f, best_grad["initstandard"], best_grad["initbalanced"]
    )

    # horizontal differential matrix
    img_f, diff_matrix = FS.get_norm_features(
        FS.get_greyscale_matrix(path),
        img_f,
        "hor",
        best_grad["horstandard"],
        best_grad["horbalanced"],
        best_pairs["hor"],
        flag=True,
    )

    # vertical differential matrix
    img_f = FS.get_norm_features(
        FS.get_greyscale_matrix(path),
        img_f,
        "vert",
        best_grad["vertstandard"],
        best_grad["vertbalanced"],
        best_pairs["vert"],
    )
    return pd.DataFrame([img_f], columns=feature_names).iloc[0], diff_matrix


def get_classification_results(parameters):
    # task_type: 1 - норма/патология, 2 - стадия фиброза
    global sensor_type, path, task_type, cur_dir, filename
    sensor_type, path, task_type = (
        parameters["sensor_type"],
        parameters["path"],
        parameters["task_type"],
    )
    cur_dir, filename = parameters["cur_dir"], parameters["filename"]

    (
        img_f,
        diff_matrix,
    ) = get_all_features()  # img_f - image features (признаки изображения)

    # МГУА
    gmdh_prob, gmdh_liver_class = calculate_gmdh_model(img_f)
    if gmdh_prob > 1 or gmdh_prob < 0:
        gmdh_prob = 100
    elif gmdh_liver_class == 2:
        gmdh_prob = round(gmdh_prob * 100, 1)
    elif gmdh_liver_class == 1:
        gmdh_prob = round((1 - gmdh_prob) * 100, 1)
    gmdh_result = "Печень в норме" if gmdh_liver_class == 1 else "Печень не в норме"

    # Лес самоорганизации
    forest_prob, forest_liver_class = forest_prediction(img_f)
    forest_result = "Печень в норме" if forest_liver_class == 1 else "Печень не в норме"

    # Пороги трёх наилучших признаков
    mean_signs = get_mean_signs(img_f)

    return (
        {
            "gmdh_result": gmdh_result,
            "gmdh_probability": gmdh_prob,
            "forest_result": forest_result,
            "forest_probability": forest_prob,
            "mean_signs": mean_signs,
        },
        diff_matrix,
    )

