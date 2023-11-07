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
def train_yourself(request, name, camera):
    # file paths
    response = {'message': 'Successfully Trained!'}
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder_path = os.path.join(BASE_DIR, 'static')
    effective_data_folder_path = os.path.join(static_folder_path, 'effective_data')
    selected_landmarks_and_combinations_path = os.path.join(effective_data_folder_path, 'selected_landmarks_and_combinations.json')
    ideal_points_path = os.path.join(static_folder_path, 'ideal_points.json')
    trained_dataset_path = os.path.join(static_folder_path, 'trained_dataset.json')
    shape_predictor_path = os.path.join(static_folder_path, 'shape_predictor_68_face_landmarks.dat')

    if os.path.isdir(static_folder_path):
        if os.path.exists(selected_landmarks_and_combinations_path):
            if os.path.exists(ideal_points_path):

                commonhelps = cmn()
                previous_trained_dataset = commonhelps.get_data_from_json_file(trained_dataset_path)
                selected_dataset = commonhelps.get_data_from_json_file(selected_landmarks_and_combinations_path)
                selected_points = selected_dataset['points']
                selected_combinations = selected_dataset['combinations']
                ideal_points = commonhelps.get_data_from_json_file(ideal_points_path)

                # name=str(input("Enter Your Name: ")).lower()
                # print(previous_trained_dataset)


                cap = cv2.VideoCapture(camera)
                hog_face_detector = dlib.get_frontal_face_detector()
                dlib_facelandmark = dlib.shape_predictor(shape_predictor_path)

                sum_of_points_value = {}
                points_to_check_side = [1, 17, 28]
                check_side_combinations = [[1, 28], [17, 28]]
                while True:
                    _, frame = cap.read()
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = hog_face_detector(gray)
                    for face in faces:
                        face_landmarks = dlib_facelandmark(gray, face)
                        xy_coordinates = {}

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
                            for point in selected_points:
                                x = face_landmarks.part(point-1).x
                                y = face_landmarks.part(point-1).y
                                xy_coordinates.update({f'x{point}': x, f'y{point}': y})
                                cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                            
                            # print(xy_coordinates)

                            ideal_point_coordinates = []
                            for ideal_point in ideal_points['elements'][ideal_points['selected']]['points']:
                                ideal_point_coordinates.append(face_landmarks.part(ideal_point-1).x)
                                ideal_point_coordinates.append(face_landmarks.part(ideal_point-1).y)

                            ideal_points_distance = commonhelps.get_distance(ideal_point_coordinates)
                            multiplier = ideal_points['elements'][ideal_points['selected']]['ideal']/ideal_points_distance

                            for selected_combination in selected_combinations:
                                if not f'{selected_combination[0]}-{selected_combination[1]}' in sum_of_points_value:
                                    sum_of_points_value.update({f'{selected_combination[0]}-{selected_combination[1]}': []})
                                distance = commonhelps.get_distance([xy_coordinates[f'x{selected_combination[0]}'], xy_coordinates[f'y{selected_combination[0]}'], xy_coordinates[f'x{selected_combination[1]}'], xy_coordinates[f'y{selected_combination[1]}']])
                                sum_of_points_value[f'{selected_combination[0]}-{selected_combination[1]}'].append(distance*multiplier)
                    cv2.imshow("Face Landmarks", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                previous_trained_dataset.update({name: [sum(sum_of_points_value[key])/len(sum_of_points_value[key]) for key in sum_of_points_value.keys()]})
                if previous_trained_dataset[name]:
                    commonhelps.create_json_file(trained_dataset_path, previous_trained_dataset)


                cap.release()
                cv2.destroyAllWindows()
            else:
                response['message'] = 'ideal_points doesn\'t exists!'
        else:
            response['message'] = 'Selected Landmarks and Combinations doesn\'t exists!'
    else:
        commonhelps.check_dir_ifnot_found_then_create(static_folder_path)
        response['message'] = 'static folder doesn\'t exists!'

    return Response(response)
