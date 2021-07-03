class CameraParameters:

    def __init__(self, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.rotation_vectors = rotation_vectors
        self.translation_vectors = translation_vectors

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
