# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

from pathlib import Path
import logging
import argparse
import abc

import cv2
import numpy as np

from objloader_simple import OBJ
from keypoint_detector import OrbKeypointDetector, SiftKeypointDetector
from keypoint_matcher import BfKeypointMatcher, FlannKeypointMatcher

# Minimum number of matches that have to be found to consider the recognition valid
MIN_MATCHES = 3
# Monocolor of the 3d AR object displayed
DEFAULT_COLOR = (50, 50, 50)

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

class Frame:
    def __init__(self, image, keypoint_detector):
        self.frame = image
        self.keypoints = None
        self.descriptors = None
        # Compute model keypoints and its descriptors
        self.keypoints, self.descriptors = keypoint_detector.detect_and_compute(image)

class ARObject(Frame):
    def __init__(self, reference_image_2d, keypoint_detector, object_3d):
        super().__init__(reference_image_2d, keypoint_detector)
        self.object = object_3d

def draw_matches(frame_from, frame_to, matches):
    """
    helper function for debugging
    """
    img1 = frame_from.frame
    keypoints1 = frame_from.keypoints
    img2 = frame_to.frame
    keypoints2 = frame_to.keypoints
    img_match = np.empty((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, outImg=img_match, matchColor=None, flags=0)
    cv2.imshow("matches", img_match)
    cv2.waitKey(0)


def find_homography(feature_matcher, frame_from, frame_to):
    # match frame descriptors with model descriptors
    try:
        matches = feature_matcher.match(frame_from.descriptors, frame_to.descriptors)
        logging.debug(f'{len(matches)} matches found')
    except cv2.error:
        logging.warn('Failed to match features')
        matches = list()

    if len(matches) > MIN_MATCHES:
        # differenciate between source points and destination points
        pts_from = np.float32([frame_from.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_to = np.float32([frame_to.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        homography, _ = cv2.findHomography(pts_from, pts_to, cv2.RANSAC, 5.0)
        return True, homography
    else:
        logging.warn(f'Not enough matches found - {len(matches)}/{MIN_MATCHES}')
        return False, None

def augment_frame(frame, ar_object, camera_parametrs, feature_matcher):
    homography_found, homography = find_homography(feature_matcher, ar_object, frame)
    if homography_found:
        frame.frame = draw_rectangle_around_found_model(frame.frame, ar_object.frame, homography)
        # if a valid homography matrix was found render cube on model plane
        try:
            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(camera_parameters, homography)
            # project cube or model
            frame = render(frame.frame, ar_object.object, projection, ar_object.frame)
        except ValueError:
            logging.warn('Failed to project object onto frame')
    else:
        logging.warn('Failed to find homography')

def draw_rectangle_around_found_model(frame, model, homography):
    # Draw a rectangle that marks the found model in the frame
    h, w, _ = model.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # project corners into frame
    try:
        dst = cv2.perspectiveTransform(pts, homography)
        # connect them with lines  
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    except cv2.error:
        logging.warn('Failed to draw bounding rectangle')
    return frame

def render(frame, obj, projection, model, scale=3, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w, _ = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        try:
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(frame, imgpts, DEFAULT_COLOR)
            else:
                color = hex_to_rgb(face[-1])
                color = color[::-1]  # reverse
                cv2.fillConvexPoly(frame, imgpts, color)
        except cv2.error:
            logging.warn('Failed to project the objecto onto frame')

    return frame

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = np.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

if __name__ == '__main__':
    # matrix of camera parameters (made up but works quite well for me); hardcoded
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    parser = argparse.ArgumentParser('REF and OBJ + --video or IMG. camera parameters are currently hardcoded')
    parser.add_argument('-r', '--ref', default='./reference/reference.jpg', help='path to the 2d reference image file', type=str)
    parser.add_argument('-o', '--obj', default='./models/fox.obj', help='path to the .obj file of a 3d object model', type=str)
    parser.add_argument('-v', '--video', action='store_true', default=False, help='if present; the webcam stream is used. Otherwise a path to an image file must be provided')
    parser.add_argument('-i', '--img', default='./reference/reference_rot.jpg', help='path to the 2d image file, e.g. rotated version of refernce', type=str)
    args = parser.parse_args()
    use_webcam_not_image = args.video

    # load the reference surface that will be searched in the video stream
    reference_image_2d_path = Path(args.ref)
    reference_image_2d_path = reference_image_2d_path.resolve()
    reference_image_2d = cv2.imread(str(reference_image_2d_path))

    # Load 3D model from OBJ file
    obj_3d_path = Path(args.obj)
    obj_3d_path = obj_3d_path.resolve()
    obj_3d = OBJ(obj_3d_path, swapyz=True)

    #keypoint_detector = OrbKeypointDetector()
    #feature_matcher = BfKeypointMatcher()
    keypoint_detector = SiftKeypointDetector()
    feature_matcher = FlannKeypointMatcher()
    
    ar_object = ARObject(reference_image_2d, keypoint_detector, obj_3d)


    if use_webcam_not_image:
        # init video captureG
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        while True:
            _, frame_image = cap.read()
            frame = Frame(frame_image, keypoint_detector)
            augment_frame(frame, ar_object, camera_parameters, feature_matcher)
            cv2.imshow('frame', frame.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        image_2d_path = Path(args.img)
        image_2d_path = image_2d_path.resolve()
        image_2d = cv2.imread(str(image_2d_path))
        frame = Frame(image_2d, keypoint_detector)
        augment_frame(frame, ar_object, camera_parameters, feature_matcher)
        cv2.imshow('frame', frame.frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass
