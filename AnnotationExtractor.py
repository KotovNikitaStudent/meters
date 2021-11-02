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

    def coordinate_convert(self):
        self._center_x = self._x_min + ((self._x_max - self._x_min) / 2)
        self._center_y = self._y_min + ((self._x_max - self._y_min) / 2)
        self._height = self._y_max - self._y_min
        self._width = self._x_max - self._x_min

    def extract_from_json(self, path_to_json: str, conver_coord=False) -> list:
        if not path_to_json.endswith('.json'):
            raise RuntimeError(f"Can not open {path_to_json.split('/')[-1]} file")
        else:
            with open(path_to_json, 'r') as file:
                data = load(file)
            file.close()

        data_from_json = []
        self._image_name = data['imagePath'][:-4]
        for obj in data['shapes']:
            self._label = obj['label']
            self._x_min = obj['points'][0][0]
            self._y_min = obj['points'][0][1]
            self._x_max = obj['points'][2][0]
            self._y_max = obj['points'][2][1]

            if conver_coord:
                self.coordinate_convert()
                data_from_json.append([self._image_name, self._label,
                                       self._center_x, self._center_y, self._width, self._height])
            else:
                data_from_json.append([self._image_name, self._label,
                                       self._x_min, self._y_min, self._x_max, self._y_max])

        return data_from_json

    def extract_from_xml(self, path_to_xml: str, conver_coord=False) -> list:
        if not path_to_xml.endswith('.xml'):
            raise RuntimeError(f"Can not open {path_to_xml.split('/')[-1]} file")
        else:
            root = ET.parse(path_to_xml).getroot()
            data_from_xml = []
            self._image_name = root.find('filename').text[:-4]

            for obj in root.findall('object'):
                self._label = obj.find('name').text
                self._x_min = float(obj.find('bndbox/xmin').text)
                self._y_min = float(obj.find('bndbox/ymin').text)
                self._x_max = float(obj.find('bndbox/xmax').text)
                self._y_max = float(obj.find('bndbox/ymax').text)

                if conver_coord:
                    self.coordinate_convert()
                    data_from_xml.append([self._image_name, self._label,
                                          self._center_x, self._center_y, self._width, self._height])
                else:
                    data_from_xml.append([self._image_name, self._label,
                                           self._x_min, self._y_min, self._x_max, self._y_max])

            return data_from_xml