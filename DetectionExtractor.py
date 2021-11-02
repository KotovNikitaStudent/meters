from json import load


class DetectionExtractor:
    def __init__(self):
        self._image_name = None
        self._label = None
        self._x_min = None
        self._y_min = None
        self._x_max = None
        self._y_max = None
        self.score = None
        self._center_x = None
        self._center_y = None
        self._width = None
        self._height = None

    def coordinate_convert(self):
        self._center_x = self._x_min + ((self._x_max - self._x_min) / 2)
        self._center_y = self._y_min + ((self._x_max - self._y_min) / 2)
        self._height = self._y_max - self._y_min
        self._width = self._x_max - self._x_min

    def extract_from_txt(self):
        pass

    def extract_from_json(self):
        pass