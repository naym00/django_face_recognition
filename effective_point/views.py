from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from helps.commom import Common as cmn
from pathlib import Path
import json
import glob
import dlib
import cv2
import os

def select_data_based_on_category(category, SELECT_BASED_ON, selected, key, dictionary, commonhelps, selected_combinations):
    if category in SELECT_BASED_ON:
        selected['data'].update({key: dictionary[key]})
        pair_point = [int(point) for point in key.split("-")]

        for pair in commonhelps.get_symmetry_points(pair_point):
            if not f'{pair[0]}-{pair[1]}' in selected_combinations:
                selected_combinations.update({f'{pair[0]}-{pair[1]}':1})
    dictionary[key]['status'] = category

@api_view(['GET'])
def effective_points(request):
    response = {'message': 'effective landmarks have been successfully generated!'}
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder_path = os.path.join(BASE_DIR, 'static')
    ideal_points_path = os.path.join(static_folder_path, 'ideal_points.json')
    config_file_path = os.path.join(static_folder_path, 'config.json')
    effective_data_folder_path = os.path.join(static_folder_path, 'effective_data')
    landmarks_status_path = os.path.join(effective_data_folder_path, 'landmarks_status.json')
    selected_landmarks_and_combinations_path = os.path.join(effective_data_folder_path, 'selected_landmarks_and_combinations.json')
    image_folder_path = os.path.join(static_folder_path, 'images')
    shape_predictor_path = os.path.join(static_folder_path, 'shape_predictor_68_face_landmarks.dat')

    if os.path.isdir(static_folder_path):
        if os.path.exists(ideal_points_path):
            if os.path.exists(config_file_path):
                commonhelps = cmn()
                ideal_points =commonhelps.get_data_from_json_file(ideal_points_path)

                settings_data = commonhelps.get_data_from_json_file(config_file_path)
                POINTS = settings_data['points']
                DIFF_TOLERANCE_TO_SELECT_POINT = settings_data['diff_tolerance_to_select_point']
                COMBINATION_COUNT = settings_data['combination_count']

                commonhelps.check_dir_ifnot_found_then_create(effective_data_folder_path)
                images = [f for f in glob.glob(image_folder_path + "/**/*", recursive=True) if not os.path.isdir(f)]

                dictionary = {}
                SELECT_BASED_ON = settings_data['select_based_on']
                selected = {'data':{}, 'points':[], 'combinations':[], 'combinations_count': 0}
                selected_points = []
                selected_combinations = {}

                hog_face_detector = dlib.get_frontal_face_detector()
                dlib_facelandmark = dlib.shape_predictor(shape_predictor_path)

                for image_path in images:
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    # dim = image.shape
                    # WIDTH = 550
                    # resized_image = cv2.resize(image, (WIDTH, int((dim[0]*WIDTH)/dim[1])))
                    
                    # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = hog_face_detector(gray)
                    for face in faces:
                        face_landmarks = dlib_facelandmark(gray, face) 

                        ideal_point_coordinates = []
                        for ideal_point in ideal_points['elements'][ideal_points['selected']]['points']:
                            ideal_point_coordinates.append(face_landmarks.part(ideal_point-1).x)
                            ideal_point_coordinates.append(face_landmarks.part(ideal_point-1).y)

                        ideal_points_distance = commonhelps.get_distance(ideal_point_coordinates)
                        multiplier = ideal_points['elements'][ideal_points['selected']]['ideal']/ideal_points_distance

                        for point in POINTS[:len(POINTS)-1]:
                            two_point_coordinates = []
                            two_point_coordinates.append(face_landmarks.part(point-1).x)
                            two_point_coordinates.append(face_landmarks.part(point-1).y)

                            for rest_point in POINTS[POINTS.index(point)+1:]:
                                if not f'{point}-{rest_point}' in dictionary:
                                    dictionary.update({f'{point}-{rest_point}':{}})

                                two_point_coordinates.append(face_landmarks.part(rest_point-1).x)
                                two_point_coordinates.append(face_landmarks.part(rest_point-1).y)

                                if not 'values' in dictionary[f'{point}-{rest_point}']:
                                    dictionary[f'{point}-{rest_point}'].update({'values':[]})
                                if not 'difference' in dictionary[f'{point}-{rest_point}']:
                                    dictionary[f'{point}-{rest_point}'].update({'difference': []})
                                if not 'status' in dictionary[f'{point}-{rest_point}']:
                                    dictionary[f'{point}-{rest_point}'].update({'status': ''})
                                if not 'average' in dictionary[f'{point}-{rest_point}']:
                                    dictionary[f'{point}-{rest_point}'].update({'average': 0})

                                dictionary[f'{point}-{rest_point}']['values'].append(commonhelps.get_distance(two_point_coordinates)*multiplier)
                                two_point_coordinates.pop()
                                two_point_coordinates.pop()
                for key in dictionary.keys():
                    store_diff = []
                    for value in dictionary[key]['values'][:len(dictionary[key]['values'])-1]:
                        for rest_value in dictionary[key]['values'][dictionary[key]['values'].index(value)+1:]:
                            store_diff.append(abs(value-rest_value))
                    dictionary[key]['difference'] = store_diff[:]

                for key in dictionary.keys():
                    average = sum(dictionary[key]['difference'])/len(dictionary[key]['difference'])
                    dictionary[key]['average'] = average
                    best = 100
                    good = 80
                    poor = 50
                    worst = 20
                    count = 0
                    for difference in dictionary[key]['difference']:
                        if difference>=DIFF_TOLERANCE_TO_SELECT_POINT:
                            count += 1
                    in_percentage = (count/len(dictionary[key]['difference']))*100

                    if in_percentage>=best:
                        select_data_based_on_category('best', SELECT_BASED_ON, selected, key, dictionary, commonhelps, selected_combinations)
                    elif in_percentage>=good:
                        select_data_based_on_category('good', SELECT_BASED_ON, selected, key, dictionary, commonhelps, selected_combinations)
                    elif in_percentage>=poor:
                        select_data_based_on_category('poor', SELECT_BASED_ON, selected, key, dictionary, commonhelps, selected_combinations)
                    else:
                        select_data_based_on_category('worst', SELECT_BASED_ON, selected, key, dictionary, commonhelps, selected_combinations)

                carry_selected_combinations = []
                for pair in selected_combinations.keys():
                    pair_point = [int(point) for point in pair.split("-")]
                    carry_selected_combinations.append(pair_point)
                
                combinations = []
                if COMBINATION_COUNT<len(carry_selected_combinations):
                    combinations = carry_selected_combinations[:COMBINATION_COUNT]
                    selected['combinations'] = combinations
                    selected['combinations_count'] = len(combinations)

                    for combination in combinations:
                        selected_points.extend(combination)
                    selected_points = list(set(selected_points))
                    selected_points.sort()
                    selected['points'] = selected_points
                else:
                    selected['combinations'] = carry_selected_combinations
                    selected['combinations_count'] = len(carry_selected_combinations)

                    for combination in carry_selected_combinations:
                        selected_points.extend(combination)
                    selected_points = list(set(selected_points))
                    selected_points.sort()
                    selected['points'] = selected_points



                if COMBINATION_COUNT<len(carry_selected_combinations):
                    selected['combinations'] = carry_selected_combinations[:COMBINATION_COUNT]
                else:
                    selected['combinations'] = carry_selected_combinations
                selected['combinations_count'] = len(selected['combinations'])

                commonhelps.remove_file(landmarks_status_path)
                commonhelps.remove_file(selected_landmarks_and_combinations_path)

                commonhelps.create_json_file(landmarks_status_path, dictionary)
                commonhelps.create_json_file(selected_landmarks_and_combinations_path, selected)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                response['message'] = 'config file is missing!'
        else:
            response['message'] = 'ideal_points doesn\'t exists!'
    else:
        commonhelps.check_dir_ifnot_found_then_create(static_folder_path)
        response['message'] = 'static folder doesn\'t exists!'

    return Response(response)
    