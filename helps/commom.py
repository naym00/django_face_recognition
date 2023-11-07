from pathlib import Path
import math
import json
import os

class Common:
    REVERSE_LANDMARKS = {"1":17,"2":16,"3":15,"4":14,"5":13,"6":12,"7":11,"8":10,"18":27,"19":26,"20":25,"21":24,"22":23,"32":36,"33":35,"37":46,"38":45,"39":44,"40":43,"41":48,"42":47,"49":55,"50":54,"51":53,"59":57,"60":56,"61":65,"62":64,"68":66}
    
    def get_data_from_json_file(self, path, blank_type={}):
        if os.path.exists(path):
            with open(path, 'r') as openfile:
                data = json.load(openfile)
            return data
        else:
            return blank_type
    
    def create_json_file(self, path, data):
        json_object = json.dumps(data)
        with open(path, "w") as outfile:
            outfile.write(json_object)
        
    def check_dir_ifnot_found_then_create(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def remove_file(self, path):
        if os.path.exists(path):
            os.remove(path)

    def get_distance(self, coordinates):
        # coordinates = [x1, y1, x2, y2]
        return math.sqrt(pow((coordinates[0]-coordinates[2]), 2)+pow((coordinates[1]-coordinates[3]), 2))
    
    def diff_check(self, val1, val2, value_to_compare=2):
        # print(f'{val1} {val2}')
        return 0 if abs(val1-val2) > value_to_compare else 1
    
    def get_diff(self, val1, val2):
        return abs(val1-val2)
    
    def get_side(self, vals_to_detect_side):
        # vals_to_detect_side = [val1, val2]
        if abs(vals_to_detect_side[0]-vals_to_detect_side[1])>12:
            if vals_to_detect_side[0] > vals_to_detect_side[1]:
                return 'left_side'
            else:
                return 'right_side'
        else:
            return 'front_side'
    

    def get_symmetry_points(self, points):
        pair = []
        pair.append(points)
        if str(points[0]) in self.REVERSE_LANDMARKS:
            if len(list({self.REVERSE_LANDMARKS[str(points[0])], points[1]})) == 2:
                pair.append([self.REVERSE_LANDMARKS[str(points[0])], points[1]])
        if str(points[1]) in self.REVERSE_LANDMARKS:
            if len(list({points[0], self.REVERSE_LANDMARKS[str(points[1])]})) == 2:
                pair.append([points[0], self.REVERSE_LANDMARKS[str(points[1])]])
        return [] if len(pair) == 3 else pair