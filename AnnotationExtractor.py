import xml.etree.ElementTree as ET
from json import load


class AnnotationExtractor:
    def __init__(self):
        self._image_name = None
        self._label = None
        self._x_min = None
        self._y_min = None
        self._x_max = None
        self._y_max = None
        self._center_x = None
        self._center_y = None
        self._width = None
        self._height = None

    def __xml_to_coco_convert(self):
        self._center_x = self._x_min + ((self._x_max - self._x_min) / 2)
        self._center_y = self._y_min + ((self._x_max - self._y_min) / 2)
        self._height = self._y_max - self._y_min
        self._width = self._x_max - self._x_min

    def __coco_to_xml_convert(self):
        self._x_min = self._center_x - (self._width / 2)
        self._x_max = self._center_x + (self._width / 2)
        self._y_min = self._center_y - (self._height / 2)
        self._y_max = self._center_y + (self._height / 2)

    def gt_coco_format(self, path_to_file):
        data_from_file = []
        if path_to_file.endswith('.json') or path_to_file.endswith('.JSON'):
            with open(path_to_file, 'r') as file:
                data = load(file)
            file.close()

            self._image_name = data['images'][0]['file_name'].split('.')[0]
            class_dict = {}

            for i in data['categories']:
                class_dict.update({i['id']: i['name']})

            for i in data['annotations']:
                self._label = class_dict[i["category_id"]]
                self._center_x = i['bbox'][0]
                self._center_y = i['bbox'][1]
                self._width = i['bbox'][2]
                self._height = i['bbox'][3]
                self.__coco_to_xml_convert()
                data_from_file.append([self._image_name, self._label,
                                      self._x_min, self._y_min, self._x_max, self._y_max])

            return data_from_file
        else:
            raise RuntimeError("File extension is incorrect")

    def gt_pascal_voc_format(self, path_to_file):
        data_from_file = []
        if path_to_file.endswith('.xml'):
            root = ET.parse(path_to_file).getroot()
            self._image_name = root.find('filename').text[:-4]

            for obj in root.findall('object'):
                self._label = obj.find('name').text
                self._x_min = float(obj.find('bndbox/xmin').text)
                self._y_min = float(obj.find('bndbox/ymin').text)
                self._x_max = float(obj.find('bndbox/xmax').text)
                self._y_max = float(obj.find('bndbox/ymax').text)
                data_from_file.append([self._image_name, self._label,
                                       self._x_min, self._y_min, self._x_max, self._y_max])

            return data_from_file
        else:
            raise RuntimeError("File extension is incorrect")

    def gt_yolo_pascal_voc_json_format(self, path_to_file):
        data_from_file = []
        if path_to_file.endswith('.json') or path_to_file.endswith('.JSON'):
            with open(path_to_file, 'r') as file:
                data = load(file)
            file.close()

            self._image_name = data['imagePath'][:-4]
            for obj in data['shapes']:
                self._label = obj['label']
                self._x_min = obj['points'][0][0]
                self._y_min = obj['points'][0][1]
                self._x_max = obj['points'][2][0]
                self._y_max = obj['points'][2][1]
                data_from_file.append([self._image_name, self._label,
                                       self._x_min, self._y_min, self._x_max, self._y_max])

            return data_from_file
        else:
            raise RuntimeError("File extension is incorrect")