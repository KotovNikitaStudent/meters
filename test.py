from DetectionExtractor import *
from AnnotationExtractor import *

ae = AnnotationExtractor()
# print(ae.gt_yolo_pascal_voc_json_format('/Users/nikita/Desktop/zip_meters/full_dataset/00_dataset/meters01/Annotations/mrsk2021_meter01_0000000000.json'))
# print(ae.gt_pascal_voc_format('/Users/nikita/Desktop/zip_meters/crop_dataset/00_dataset/meters01/Annotations/mrsk2021_meter01_0000000000.xml'))
# print(ae.gt_coco_format('/Users/nikita/Desktop/zip_meters/toy_example_coco.JSON'))

# print(ae.extract_data('/Users/nikita/Desktop/zip_meters/crop_dataset/00_dataset/meters01/Annotations/mrsk2021_meter01_0000000000.xml'))
# print(ae.extract_data('/Users/nikita/Desktop/zip_meters/full_dataset/00_dataset/meters01/Annotations/mrsk2021_meter01_0000000000.json'))

de = DetectionExtractor()
de.det_class_name_x1y1x2y2('/Users/nikita/Desktop/zip_meters/result_ed/ed0/05/mrsk2021_meter01_0000000002.txt')
# print(de.extract_data('/Users/nikita/Desktop/zip_meters/results_yolov4/results_meters_320/comp4_det_test_breaker.txt'))
