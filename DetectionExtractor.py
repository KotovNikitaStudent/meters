from json import load


classes = {value['id']: value['name'] for value in load(open('classes.json', 'r')).values()}


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

    def __coordinate_convert(self):
        self._center_x = self._x_min + ((self._x_max - self._x_min) / 2)
        self._center_y = self._y_min + ((self._x_max - self._y_min) / 2)
        self._height = self._y_max - self._y_min
        self._width = self._x_max - self._x_min

    def extract_data(self, path_to_file: str, convert_coord=False):
        data_from_file = []
        if path_to_file.endswith('.txt'):
            self._label = path_to_file.split('/')[-1].split('.')[0].split('_')[-1]
            with open(path_to_file, 'r') as file:
                for line in file:
                    split_data = line.strip().split(' ')
                    self._image_name = split_data[0]
                    self.score = float(split_data[1])
                    self._x_min = float(split_data[2])
                    self._y_min = float(split_data[3])
                    self._x_max = float(split_data[4])
                    self._y_max = float(split_data[5])

                    if convert_coord:
                        self.__coordinate_convert()
                        data_from_file.append([self._image_name, self._label,
                                              self._center_x, self._center_y, self._width, self._height,
                                              self.score])
                    else:
                        data_from_file.append([self._image_name, self._label,
                                              self._x_min, self._y_min, self._x_max, self._y_max,
                                              self.score])

            return data_from_file
        elif path_to_file.endswith('.json'):
            pass
        else:
            raise RuntimeError("File extension is incorrect")