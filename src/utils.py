import numpy as np


NOSE = 0
R_EYE = 15
L_EYE = 16
R_EAR = 17
L_EAR = 18

HEAD_IDS = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR]

R_SHOULDER_ID = 2
L_SHOULDER_ID = 5
SHOULDER_IDS = [L_SHOULDER_ID, R_SHOULDER_ID]

THRESHOLD = 0.05


def get_face_pts(keypoints):
    """Extract the keypoints belonging to the head,
    only if the confidence is greater than the threshold
    """
    return keypoints[HEAD_IDS, :2][keypoints[HEAD_IDS, -1] > THRESHOLD]


def get_head_bbox(keypoints):
    """This function is adapted from the PeopleTracker module of Gaze360.
    It extracts face keypoints and calculates the min and max coordinates
    of an initial bounding box.
    It then uses shoulder keypoints to find the neck and enlarge the bbox.
    Finally, it finds the center and generate a square box.
    """
    face_pts = get_face_pts(keypoints)

    if not np.any(face_pts):
        return None

    face_A = np.min(face_pts, axis=0)
    face_B = np.max(face_pts, axis=0)
    face_W = face_B[0] - face_A[0]
    face_mid = (face_A + face_B) / 2

    neck_mask = keypoints[SHOULDER_IDS, -1] > THRESHOLD
    if np.any(neck_mask):
        neck = np.mean(keypoints[SHOULDER_IDS, :2][neck_mask, :], axis=0)
    else:
        neck = face_mid + [0, face_W * 0.6]

    neck_H = abs(neck[1] - face_mid[1])
    head_A = face_mid - [face_W * 0.5 * 1.5, neck_H * 1.0]
    head_B = face_mid + [face_W * 0.5 * 1.5, neck_H * 1.0]

    # make square
    bbox_size = np.max(head_B - head_A)
    bbox_center = (head_A + head_B) / 2
    bbox_A = bbox_center - bbox_size * 0.5
    bbox_B = bbox_center + bbox_size * 0.5
    head_bbox = np.concatenate((bbox_A, bbox_B - bbox_A))

    return head_bbox


def extract_all_faces(people):
    """For every person detected in a frame extract face points and
    head bbox.
    """

    faces, heads = [], []

    for p in people:
        keypoints = p["pose_keypoints_2d"]

        # keypoints is a list of (x, y, confidence)
        # reshape to (N_people, 3)
        keypoints = np.reshape(keypoints, (-1, 3))

        face = get_face_pts(keypoints)
        head = get_head_bbox(keypoints)

        if face is not None and head is not None:
            faces.append(face)
            heads.append(head)

    return faces, heads


def compute_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    eps = 1e-8

    if iou <= 0.0 or iou > 1.0 + eps:
        return 0.0

    return iou


def pointsize_to_pointpoint(bbox):
    """Function that changes bbox representation from
    (x, y, w, h) to 
    (x1, y1, x2, y2)
    """
    x_left, y_top = bbox[:2]
    w, h = bbox[2:]

    x_right = x_left + w
    y_bottom = y_top + h

    bbox = np.array([x_left, y_top, x_right, y_bottom])

    return bbox

