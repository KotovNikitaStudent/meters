class PreprocessData:
    def __init__(self, class_dict, data_list):
        self._class_dict = class_dict
        self._data_list = data_list

    def calculate_objects(self):
        if not isinstance(self._data_list, list):
            raise ValueError('Object is not a list')
        else:
            pass

    def preprocess(self):
        if not isinstance(self._data_list, list):
            raise ValueError('Object is not a list')
        else:
            pass
