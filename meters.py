import os
import json
import xml.etree.ElementTree as ET
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.compute_overlap import compute_overlap
from termcolor import colored


# SUBFOLDER_FOR_YOLO = 'results_yolov4'
SUBFOLDER_FOR_YOLO = 'results_8684'

PATH_TO_FOLDER_WITH_XML = '/Users/nikita/PycharmProjects/efficientdet/efficientdet/efficientdet/00_dataset/meters01/Annotations/'

PATH_TO_FOLDER_WITH_JSON = '/Users/nikita/Desktop/full_dataset/meters01/Annotations/'
# PATH_TO_LIST_OF_IMAGES_JSON = '/Users/nikita/Desktop/full_dataset/meters01/ImageSets/Main/test.txt'
PATH_TO_LIST_OF_IMAGES_JSON = '00_dataset/meters01/ImageSets/Main/test.txt'

PATH_TO_FOLDER_WITH_TXT = '/Users/nikita/Desktop/detections/ed4/05/'
PATH_TO_LIST_OF_IMAGES = '00_dataset/meters01/ImageSets/Main/test.txt'
PATH_TO_JSON_WITH_RESULTS_OF_DETECTIONS = '/Users/nikita/Desktop/test_ssd/meters_data/test_ssd512_sgdc.json'
PATH_WITH_DATA_FOR_DRAW_CURVE = '/Users/nikita/PycharmProjects/efficientdet/efficientdet/efficientdet/test_ssd_write.json'
# PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO = '/Users/nikita/Desktop/results_yolov4/results_tiny3l_meters_extra'
PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO = '/Users/nikita/Desktop/results_8684/results_yolov4-tiny-fake_8684'
# PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO = '/Users/nikita/Desktop/results_fake/ppyolo_v2_r50/results_fake_320'

class_name = ['meter', 'value', 'seal', 'mag', 'seal2', 'model', 'serial', 'breaker']
label_number = {'meter': 0, 'value': 1, 'seal': 2, 'mag': 3, 'seal2': 4, 'model': 5, 'serial': 6, 'breaker': 7}

NAME_PLOT = PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO.split('/')[-1] # yolo
# NAME_PLOT = PATH_TO_JSON_WITH_RESULTS_OF_DETECTIONS.split('/')[-1][:-5] # ssd
# NAME_PLOT = PATH_TO_FOLDER_WITH_TXT.split('/')[-3]# efficientdet


def main():
    # list_of_files = []
    # with open(PATH_TO_LIST_OF_IMAGES, 'r') as file:
    #     for line in file:
    #         list_of_files.append(line.strip())
    # file.close()

    list_of_files = []
    with open(PATH_TO_LIST_OF_IMAGES_JSON, 'r') as file:
        for line in file:
            list_of_files.append(line.strip())
    file.close()

    # list_of_files = sorted(list_of_files)

    model = 'yolo' # 'efficientdet', 'ssd', 'yolo'

    if model == 'efficientdet':
        prepare_data_for_test_efficientdet(list_of_files)
    if model == 'ssd':
        prepare_data_for_test_ssd(list_of_files)
    if model == 'yolo':
        prepare_data_for_test_yolo(list_of_files, PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO)

    get_metrics_and_figure(NAME_PLOT, show_fig=True, save_fig=True, write_to_csv_file=False, write_to_terminal=False)


def get_data_for_metrics(det_arr: list, ann_arr: list) -> None:
    """calculation of F1, R, P, score metrics, recording data about them in the .json file"""
    all_detections = det_arr
    all_annotations = ann_arr

    pr, rc, f1_m, sc, ap = [], [], [], [], []
    average_precisions = 0
    iou_threshold = 0.5

    for label in range(len(all_detections[0])):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0

        for i in range(len(all_detections)):

            if len(all_annotations[i][label]) != 0:
                annotations = all_annotations[i][label]
            else:
                continue

            if len(all_detections[i][label]) != 0:
                detections = all_detections[i][label]
            else:
                continue

            num_annotations += len(annotations)
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                d = np.asarray([d])
                annotations = np.array([np.array(xi) for xi in annotations], dtype=np.float64)

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(d, annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions = 0
            continue

        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        pr.append(precision.tolist())
        rc.append(recall.tolist())

        f1 = 2.0 * (precision * recall) / (precision + recall + 1e-9)
        f1_m.append(f1.tolist())
        sc.append(np.sort(scores).tolist())
        average_precisions = compute_ap(recall, precision)
        ap.append(average_precisions)

    a = {}
    a.update({"pr": pr, "rc": rc, "sc": sc, "f1_m": f1_m, "ap": ap})
    with open(PATH_WITH_DATA_FOR_DRAW_CURVE, 'w') as fl:
        fl.write(json.dumps(a))
    fl.close()


def compute_ap(recall: list, precision: list):
    """compute the average precision, given the recall and precision curves"""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

    return ap.tolist()


def prepare_data_for_test_ssd(list_of_files: list) -> None:
    """preparation of data from testing SSD"""
    ann_arr = []
    for file in list_of_files[:]:
        ann_file = get_annotation_xml(PATH_TO_FOLDER_WITH_XML + file + '.xml')
        ann_file.sort(key=lambda x: x[0])
        class_arr = [[] for _ in range(len(class_name))]

        for i in ann_file:
            class_arr[label_number[i[1]]].append([i[2], i[3], i[4], i[5]])

        ann_arr.append(class_arr)

    det = pasre_json_ssd(PATH_TO_JSON_WITH_RESULTS_OF_DETECTIONS)
    det_arr = sort_detections_for_ssd_yolo(det, list_of_files, class_name)
    get_data_for_metrics(det_arr, ann_arr)


def prepare_data_for_test_yolo(list_of_files: list, path_to_folder_with_results_for_yolo: str, set_yolo='origin') -> None:
    """preparation of data from testing YOLO"""
    ann_arr = []
    for file in list_of_files[:]:
        ann_file = get_annotation_xml(PATH_TO_FOLDER_WITH_XML + file + '.xml')
        # ann_file = get_annotation_json(PATH_TO_FOLDER_WITH_JSON + file + '.json')
        # ann_file.sort(key=lambda x: x[0])
        class_arr = [[] for _ in range(len(class_name))]

        for i in ann_file:
            class_arr[label_number[i[1]]].append([i[2], i[3], i[4], i[5]])

        ann_arr.append(class_arr)

    # list_file_with_classes_for_yolo = []
    # for file_ in os.listdir(path_to_folder_with_results_for_yolo):
    #     if file_.endswith('.txt'):
    #         list_file_with_classes_for_yolo.append(path_to_folder_with_results_for_yolo + '/' + file_)

    class_arr_yolo = [[] for _ in range(len(class_name))]

    if set_yolo == 'origin':
        list_file_with_classes_for_yolo = [f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_meter.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_value.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_seal.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_mag.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_seal2.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_model.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_serial.txt',
                                           f'/Users/nikita/Desktop/{SUBFOLDER_FOR_YOLO}/{NAME_PLOT}/comp4_det_test_breaker.txt']
    elif set_yolo == 'fake':
        list_file_with_classes_for_yolo = [f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_meter.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_value.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_seal.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_mag.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_seal2.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_model.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_serial.txt',
                                           f'{PATH_TO_FOLDER_WITH_RESULTS_FOR_YOLO}/comp4_det_test_breaker.txt']

    for i in range(len(class_arr_yolo)):
        with open(list_file_with_classes_for_yolo[i], 'r') as file:
            for line in file:
                list_data = line.strip().split(' ')
                list_data[1] = float(list_data[1])
                list_data[2] = float(list_data[2])
                list_data[3] = float(list_data[3])
                list_data[4] = float(list_data[4])
                list_data[5] = float(list_data[5])
                class_arr_yolo[i].append(list_data)

    for i in range(len(class_arr_yolo)):
        class_arr_yolo[i].sort(key=lambda x: x[0])

    det_arr = sort_detections_for_ssd_yolo(class_arr_yolo, list_of_files, class_name)
    get_data_for_metrics(det_arr[:223], ann_arr[:223])


def prepare_data_for_test_efficientdet(list_of_files: list) -> None:
    """preparation of data from testing EfficientDet"""
    ann_arr = []
    det_arr = []

    for file in list_of_files[:]:
        ann_file = get_annotation_xml(PATH_TO_FOLDER_WITH_XML + file + '.xml')
        det_file = get_detection_txt(PATH_TO_FOLDER_WITH_TXT + file + '.txt')
        ann_file.sort(key = lambda x: x[0])
        det_file.sort(key = lambda x: x[0])

        class_arr_ann = [[] for _ in range(len(class_name))]
        class_arr_det = [[] for _ in range(len(class_name))]

        for i in ann_file:
            class_arr_ann[label_number[i[1]]].append([i[2], i[3], i[4], i[5]])

        for i in det_file:
            class_arr_det[label_number[i[1]]].append([i[3], i[4], i[5], i[6], i[2]])

        ann_arr.append(class_arr_ann)
        det_arr.append(class_arr_det)

    get_data_for_metrics(det_arr, ann_arr)


def get_metrics_and_figure(name_plot: str, save_fig=False, show_fig=False, write_to_terminal=False, write_to_csv_file=False) -> None:
    """draw curves of dependence of Precision(Recall) and F1(Threshold), obtaining Precision, Recall, F1"""
    with open(PATH_WITH_DATA_FOR_DRAW_CURVE, 'r') as f:
        data = json.loads(f.read())

    precision = data['pr']
    recall = data['rc']
    score = data['sc']
    f1 = data['f1_m']
    ap = data['ap']

    if not os.path.exists('./draw'):
        os.mkdir('./draw')

    # class_name_1 = ['meter', 'value', 'seal', 'mag', 'seal2', 'model', 'serial', 'breaker']
    # colors = ['purple', 'sienna', 'green', 'orange', 'gray', 'r', 'violet', 'blue']

    class_name_1 = ['mag', 'model', 'meter', 'value', 'seal2', 'seal', 'serial', 'breaker']
    colors = ['purple', 'sienna', 'green', 'orange', 'gray', 'r', 'violet', 'blue']

    for p, r, class_, label, col in zip(precision, recall, class_name, class_name_1, colors):
        plt.plot(r, p, label=label, color=col)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs. Recall curve')
        plt.grid(True)
    plt.legend()
    if save_fig:
        plt.savefig(f'draw/{name_plot}_RP.jpg')
    if show_fig:
        plt.show()

    for f, sc, class_, label, col in zip(f1, score, class_name, class_name_1, colors):
        plt.plot(sc, f, label=label, color=col)
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title('F1(Threshold) curve')
        plt.grid(True)
    plt.legend()
    if save_fig:
        plt.savefig(f'draw/{name_plot}_F1.jpg')
    if show_fig:
        plt.show()

    precisions_list = [truncate(i[np.argmax(i)], 4) for i in precision]
    recall_list = [truncate(i[np.argmax(i)], 4) for i in recall]
    f1_list = [truncate(i[np.argmax(i)], 4) for i in f1]
    ap_list = [truncate(ap[i], 4) for i in range(len(ap))]

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from beautifultable import BeautifulTable
    table = BeautifulTable()

    table.column_headers = [f"{colored('Precision:', 'blue', attrs=['bold'])}",
                            f"{colored('Recall:', 'blue', attrs=['bold'])}",
                            f"{colored('F1:', 'blue', attrs=['bold'])}",
                            f"{colored('AP:', 'blue', attrs=['bold'])}"]

    if write_to_terminal:
        for i, j, k, z in zip(f1, precision, recall, range(len(ap))):
            index = np.argmax(i)
            table.append_row([truncate(j[index], 4),
                              truncate(k[index], 4),
                              truncate(2 * j[index] * k[index] / (k[index] + j[index]), 4),
                              truncate(ap[z], 4)])
        print(table)
        print(f"{colored('mAP:', 'red', attrs=['bold'])} {truncate(sum(ap)/len(ap), 4)}")

    if write_to_csv_file:
        write_meters_to_csv(class_name, precisions_list, recall_list, f1_list, ap_list)


def sort_detections_for_ssd_yolo(det: list, list_of_files: list, class_name: list) -> list:
    """assembly of detection data in the form required to obtain metrics"""
    temp = []
    for image in list_of_files:
        class_arr = [[] for _ in range(len(class_name))]
        for class_ in range(len(det)):
            for i in det[class_]:
                if i[0] == image:
                    class_arr[class_].append([i[2], i[3], i[4], i[5], i[1]])
        temp.append(class_arr)
    return temp


def pasre_json_ssd(path_to_json_file: str) -> list:
    """collecting detection data from a json file into a list"""
    with open(path_to_json_file, 'r') as f:
        data = json.loads(f.read())
    data1 = data['data']
    f.close()
    for nc in range(len(data1)):
        if len(data1[nc]) != 0:
            for j in range(len(data1[nc])):
                data1[nc][j][1] = float(data1[nc][j][1])
                data1[nc][j][2] = float(data1[nc][j][2])
                data1[nc][j][3] = float(data1[nc][j][3])
                data1[nc][j][4] = float(data1[nc][j][4])
                data1[nc][j][5] = float(data1[nc][j][5])
    return data1[1:]


def get_detection_txt(path_to_txt: str) -> list:
    """get data from txt file"""
    data_from_txt = []
    with open(path_to_txt, 'r') as f:
        for line in f:
            list_data = line.strip().split(' ')
            list_data[1] = float(list_data[1])
            list_data[2:] = map(int, list_data[2:])
            list_data.insert(0, path_to_txt.split('/')[-1][:-4])
            data_from_txt.append(list_data)
    f.close()
    return data_from_txt


def get_annotation_xml(path_to_xml: str) -> list:
    """get data from xml file """
    root = ET.parse(path_to_xml).getroot()
    data_from_xml = []
    filename = root.find('filename').text[:-4]
    for obj in root.findall('object'):
        name_obj = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        data_from_xml.append([filename, name_obj, xmin, ymin, xmax, ymax])

    return data_from_xml


def truncate(number: float, digits: int) -> float:
    """truncating a number to n decimal places"""
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def get_annotation_json(path_to_json: str) -> list:
    data_from_json = []
    with open(path_to_json, 'r') as file:
        data = json.load(file)
    file.close()

    image_name = data['imagePath'][:-4]
    for obj in data['shapes']:
        name_obj = obj['label']
        xmin = obj['points'][0][0]
        ymin = obj['points'][0][1]
        xmax = obj['points'][2][0]
        ymax = obj['points'][2][1]
        data_from_json.append([image_name, name_obj, xmin, ymin, xmax, ymax])

    return data_from_json


def get_unique_numbers(numbers):
    list_of_unique_numbers = []
    unique_numbers = set(numbers)

    for number in unique_numbers:
        list_of_unique_numbers.append(number)

    return list_of_unique_numbers


def write_meters_to_csv(*args):
    import pandas as pd
    frame = pd.DataFrame({'Meters': args[0],
                          "precision": args[1],
                          "recall": args[2],
                          "f1": args[3],
                          "ap": args[4]})

    if not os.path.exists('./meters_csv'):
        os.mkdir('./meters_csv')

    frame.to_csv(f"meters_csv/{NAME_PLOT}.csv", sep=';', index=False, index_label=True)


if __name__ == '__main__':
    main()