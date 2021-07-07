import sys
from unittest import TestCase

sys.path.append('../camera')
sys.path.append('../visualization')

from camera.landmarks_calibration import *
from camera.automatic_calibration import *
from visualization.utils import *

VIDEO_PATH = "input/videos/goodmans_yard.mp4"
LANDMARK_PATH = "input/landmarks/goodmans_yard_landmarks.csv"
ROAD_PLANE_GRID = "input/landmarks/road_plane_grid.csv"
FRAME_PATH = "input/frames/goodmans_yard_frame.jpg"
OUTPUT_PATH = "results"


class Test(TestCase):
    def test_landmarks_camera_calibration(self):
        """
        Test camera calibration by displaying a grid on the road plane on the first frame of the video.
        """
        camera_parameters = landmarks_camera_calibration(VIDEO_PATH, LANDMARK_PATH)

        print("Camera Matrix:\n", camera_parameters.camera_matrix)
        print("Distortion Coefficients: ", camera_parameters.distortion_coefficients)
        print("Rotation Vectors:\n", camera_parameters.rotation_vectors)
        print("Translation Vectors:\n ", camera_parameters.translation_vectors)

        video_capture = cv2.VideoCapture(VIDEO_PATH)

        # Get first frame of input video
        success, frame = video_capture.read()

        # Display distorted and undistorted frame to test camera calibration
        undistorted_frame = cv2.undistort(
            frame, camera_parameters.camera_matrix, camera_parameters.distortion_coefficients, None)
        cv2.imshow("distorted %s" % (frame.shape,), frame)
        cv2.imshow("undistorted %s" % (undistorted_frame.shape,), undistorted_frame)
        cv2.waitKey()

        # Fetch road plane grid coordinates from csv
        grid_points = csv.reader(open(ROAD_PLANE_GRID, 'r'), delimiter=",")

        road_plane_grid_points = []
        for row in grid_points:
            road_plane_grid_points.append(np.asarray([(row[0], row[1], row[2])], dtype='float64'))

        # Convert input to object points (required by opencv)
        road_plane_grid_object_points = np.zeros([1, len(road_plane_grid_points), 3], np.float32)
        road_plane_grid_object_points[0, :, :3] = np.asarray(road_plane_grid_points, dtype='float64').reshape(-1, 3)

        pixel_grid_object_points_list = cv2.projectPoints(road_plane_grid_object_points,
                                                          np.asarray(camera_parameters.rotation_vectors),
                                                          np.asarray(camera_parameters.translation_vectors),
                                                          np.asarray(camera_parameters.camera_matrix),
                                                          np.asarray(camera_parameters.distortion_coefficients))
        pixel_grid_object_points = pixel_grid_object_points_list[0]

        display_grid(frame, pixel_grid_object_points)

    def test_automatic_camera_calibration(self):

        camera_parameters = landmarks_camera_calibration(
            VIDEO_PATH, LANDMARK_PATH)

        previous_translation_vectors = camera_parameters.translation_vectors

        camera_parameters = automatic_camera_calibration(FRAME_PATH, OUTPUT_PATH,
                                                         camera_parameters.camera_matrix,
                                                         camera_parameters.distortion_coefficients)

        video_capture = cv2.VideoCapture(VIDEO_PATH)

        # Get first frame of input video
        success, frame = video_capture.read()

        # Fetch road plane grid coordinates from csv
        grid_points = csv.reader(open(ROAD_PLANE_GRID, 'r'), delimiter=",")

        road_plane_grid_points = []
        for row in grid_points:
            road_plane_grid_points.append(np.asarray([(row[0], row[1], row[2])], dtype='float64'))

        # Convert input to object points (required by opencv)
        road_plane_grid_object_points = np.zeros([1, len(road_plane_grid_points), 3], np.float32)
        road_plane_grid_object_points[0, :, :3] = np.asarray(road_plane_grid_points, dtype='float64').reshape(-1, 3)

        print(previous_translation_vectors)
        print(camera_parameters.translation_vectors)

        pixel_grid_object_points_list = cv2.projectPoints(road_plane_grid_object_points,
                                                          np.asarray(camera_parameters.rotation_vectors),
                                                          np.asarray(camera_parameters.translation_vectors),
                                                          np.asarray(camera_parameters.camera_matrix),
                                                          np.asarray(camera_parameters.distortion_coefficients))
        pixel_grid_object_points = pixel_grid_object_points_list[0]

        display_grid(frame, pixel_grid_object_points)

    def test_back_projection(self):

        camera_parameters = landmarks_camera_calibration(VIDEO_PATH, LANDMARK_PATH)

        video_capture = cv2.VideoCapture(VIDEO_PATH)

        # Get first frame of input video
        success, frame = video_capture.read()

        # Fetch road plane grid coordinates from csv
        grid_points_file = open(ROAD_PLANE_GRID, 'r')
        grid_points = csv.reader(grid_points_file, delimiter=",")

        road_plane_grid_points = []
        for row in grid_points:
            road_plane_grid_points.append(np.asarray([(row[0], row[1], row[2])], dtype='float64'))

        # Convert input to object points (required by opencv)
        road_plane_grid_object_points = np.zeros([1, len(road_plane_grid_points), 3], np.float32)
        road_plane_grid_object_points[0, :, :3] = np.asarray(road_plane_grid_points, dtype='float64').reshape(-1, 3)

        pixel_grid_object_points_list = cv2.projectPoints(road_plane_grid_object_points,
                                                          np.asarray(camera_parameters.rotation_vectors),
                                                          np.asarray(camera_parameters.translation_vectors),
                                                          np.asarray(camera_parameters.camera_matrix),
                                                          np.asarray(camera_parameters.distortion_coefficients))
        pixel_grid_object_points = pixel_grid_object_points_list[0]

        back_projected_world_grid_points = []
        for pixel_grid_object_point in pixel_grid_object_points:
            pixel_grid_point = pixel_grid_object_point[0]
            back_projected_world_grid_point = back_projection(pixel_grid_point, camera_parameters)
            back_projected_world_grid_points.append(back_projected_world_grid_point)
            print("Image Point: ", pixel_grid_point)
            print("World Point: ", back_projected_world_grid_point[0], back_projected_world_grid_point[1])

        grid_points_file.close()
