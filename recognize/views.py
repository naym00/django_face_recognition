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

@api_view(['GET'])
def recognize_yourself(request, camera):
    response = {'message': 'Closed!'}

    # file paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder_path = os.path.join(BASE_DIR, 'static')

    trained_dataset_path = os.path.join(static_folder_path, 'trained_dataset.json')
    effective_data_folder_path = os.path.join(static_folder_path, 'effective_data')
    selected_landmarks_and_combinations_path = os.path.join(effective_data_folder_path, 'selected_landmarks_and_combinations.json')
    ideal_points_path = os.path.join(static_folder_path, 'ideal_points.json')
    shape_predictor_path = os.path.join(static_folder_path, 'shape_predictor_68_face_landmarks.dat')
    config_file_path = os.path.join(static_folder_path, 'config.json')

    if os.path.exists(trained_dataset_path):
        if os.path.exists(selected_landmarks_and_combinations_path):
            if os.path.exists(ideal_points_path):
                if os.path.exists(config_file_path):

                    commonhelps = cmn()
                    trained_data = commonhelps.get_data_from_json_file(trained_dataset_path)
                    names = list(trained_data.keys())
                    selected_dataset = commonhelps.get_data_from_json_file(selected_landmarks_and_combinations_path)
                    selected_points = selected_dataset['points']
                    selected_combinations = selected_dataset['combinations']
                    ideal_points = commonhelps.get_data_from_json_file(ideal_points_path)
                    settings_data = commonhelps.get_data_from_json_file(config_file_path)


                    cap = cv2.VideoCapture(camera)
                    hog_face_detector = dlib.get_frontal_face_detector()
                    dlib_facelandmark = dlib.shape_predictor(shape_predictor_path)

                    LEN_SIZE_TO_DETECT_PERSON = settings_data['len_size_to_detect_person']
                    points_to_check_side = [1, 17, 28]
                    check_side_combinations = [[1, 28], [17, 28]]
                    person = []
                    while True:
                        _, frame = cap.read()
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = hog_face_detector(gray)
                        for face in faces:
                            face_landmarks = dlib_facelandmark(gray, face)

                            face_vals_to_check_side = []
                            xy_coordinates_to_check_side = {}
                            for point in points_to_check_side:
                                x = face_landmarks.part(point-1).x
                                y = face_landmarks.part(point-1).y
                                xy_coordinates_to_check_side.update({f'x{point}': x, f'y{point}': y})
                                # cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

                            for selected_combination in check_side_combinations:
                                distance = commonhelps.get_distance([xy_coordinates_to_check_side[f'x{selected_combination[0]}'], xy_coordinates_to_check_side[f'y{selected_combination[0]}'], xy_coordinates_to_check_side[f'x{selected_combination[1]}'], xy_coordinates_to_check_side[f'y{selected_combination[1]}']])
                                face_vals_to_check_side.append(distance)
                            
                            view = commonhelps.get_side(face_vals_to_check_side)
                            if view == 'front_side':
                                face_vals = []
                                xy_coordinates = {}

                                for point in selected_points:
                                    x = face_landmarks.part(point-1).x
                                    y = face_landmarks.part(point-1).y
                                    xy_coordinates.update({f'x{point}': x, f'y{point}': y})
                                    # cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

                                ideal_point_coordinates = []
                                for ideal_point in ideal_points['elements'][ideal_points['selected']]['points']:
                                    ideal_point_coordinates.append(face_landmarks.part(ideal_point-1).x)
                                    ideal_point_coordinates.append(face_landmarks.part(ideal_point-1).y)

                                ideal_points_distance = commonhelps.get_distance(ideal_point_coordinates)
                                multiplier = ideal_points['elements'][ideal_points['selected']]['ideal']/ideal_points_distance

                                for selected_combination in selected_combinations:
                                    distance = commonhelps.get_distance([xy_coordinates[f'x{selected_combination[0]}'], xy_coordinates[f'y{selected_combination[0]}'], xy_coordinates[f'x{selected_combination[1]}'], xy_coordinates[f'y{selected_combination[1]}']])
                                    face_vals.append(distance*multiplier)

                                cus_dic = {}
                                for name in names:
                                    diffs = []
                                    for index in range(len(trained_data[name])):
                                        diffs.append(commonhelps.get_diff(trained_data[name][index], face_vals[index]))
                                    cus_dic.update({f'{sum(diffs)/len(diffs)}': name})
                                avgs = [float(key) for key in cus_dic.keys()]
                                avgs.sort()
                                if avgs[0]<4.5:
                                    person.append(cus_dic[f'{avgs[0]}'])
                                elif avgs[0]>5.2:
                                    person.append('Unknown')

                        if len(person) >= 8:
                            flag = False
                            for name in names:
                                if person.count(name)>=5:
                                    print(name)
                                    flag = True
                                    break
                            # if not flag:
                            #     print('Unknown')
                            person = []

                        cv2.imshow("Face Landmarks", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    cap.release()
                    cv2.destroyAllWindows()
                else:
                    response['message'] = 'config file is missing!'
            else:
                response['message'] = 'ideal_points doesn\'t exists!'
        else:
            response['message'] = f'{selected_landmarks_and_combinations_path} file is missing!'
    else:
        response['message'] = 'Trained Dataset doesn\'t exists!'
    return Response(response)