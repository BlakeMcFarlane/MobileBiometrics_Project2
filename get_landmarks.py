import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os

def distances(points):
    dist = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            p1 = points[i,:]
            p2 = points[j,:]      
            dist.append( math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) )
    return dist

def get_bounding_box(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) 
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y 
	return x, y, w, h

def shape_to_np(shape, num_coords, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((num_coords, 2), dtype=dtype)
 	# loop over the facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, num_coords):
		coords[i] = (shape.part(i).x, shape.part(i).y) 
	# return the list of (x, y)-coordinates
	return coords

def get_landmarks(images, labels, save_directory="", num_coords=68, to_save=False):
    print("Getting %d facial landmarks" % num_coords)
    landmarks = []
    new_labels = []
    img_ct = 0

    if num_coords == 5:
        predictor_path = './shape_predictor_5_face_landmarks.dat'
    else:
        predictor_path = './shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img, label in zip(images, labels):
        img_ct += 1
        detected_faces = detector(img, 1)
        for d in detected_faces:
            new_labels.append(label)
            x, y, w, h = get_bounding_box(d)
            points = shape_to_np(predictor(img, d), num_coords)

            # Compute proportions
            proportions = compute_proportions(points)
            landmarks.append(proportions)
            
            # Calculate distances from centroid
            centroid = np.mean(points, axis=0)

            if to_save:
                # Create a copy of the image to draw on
                img_copy = img.copy()
                
                # Draw lines from the centroid to each landmark
                for (x_, y_) in points:
                    # Draw the line
                    cv2.line(img_copy, (int(centroid[0]), int(centroid[1])), (x_, y_), (255, 0, 0), 1)
                    # Draw the landmark point
                    cv2.circle(img_copy, (x_, y_), 2, (0, 255, 0), -1)
                
                # Draw the centroid
                cv2.circle(img_copy, (int(centroid[0]), int(centroid[1])), 3, (0, 0, 255), -1)

                # Convert BGR to RGB for correct color display in Matplotlib
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

                # Save the image using Matplotlib
                plt.figure()
                plt.imshow(img_copy)
                if not os.path.isdir(save_directory):
                    os.mkdir(save_directory)
                plt.savefig(os.path.join(save_directory, f"{label}_{img_ct}.png"), dpi=300, bbox_inches="tight")
                plt.close()

            if img_ct % 50 == 0:
                print(f"{img_ct} images with facial landmarks completed.")

    print("Landmarks shape:", np.array(landmarks).shape)
    print("Labels:", len(new_labels))

    return np.array(landmarks), np.array(new_labels)


            
            
def distances_from_centroid(points):
    """
    Compute distances from the centroid of the face to each facial landmark.
    
    Args:
        points (numpy array): Array of shape (68, 2) containing facial landmarks' coordinates.
    
    Returns:
        list: Distances from the centroid to each of the 68 landmarks.
    """
    # Calculate the centroid of the landmarks
    centroid = np.mean(points, axis=0)
    
    # Calculate the Euclidean distance from the centroid to each landmark
    dist = [math.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2) for p in points]
    
    return dist


import numpy as np
import math
from scipy.spatial import ConvexHull

import numpy as np
import math
from scipy.spatial import ConvexHull

def compute_proportions(points):
    """
    Compute a variety of facial proportions based on facial landmarks.

    Args:
        points (numpy array): Array of shape (68, 2) containing facial landmarks.

    Returns:
        list: Proportions computed from various facial landmarks.
    """
    # Initialize proportions list
    proportions = []
    epsilon = 1e-6  # Small constant to prevent division by zero

    # Compute basic distances
    # Face width (jawline width)
    face_width = np.linalg.norm(points[16] - points[0]) + epsilon

    # Face height (from chin to nose bridge)
    face_height = np.linalg.norm(points[8] - points[27]) + epsilon

    # Nose length
    nose_length = np.linalg.norm(points[27] - points[33]) + epsilon

    # Nose width (nostril to nostril)
    nose_width = np.linalg.norm(points[35] - points[31]) + epsilon

    # Mouth width (corner to corner)
    mouth_width = np.linalg.norm(points[54] - points[48]) + epsilon

    # Mouth height (upper to lower lip)
    mouth_height = np.linalg.norm(points[66] - points[62]) + epsilon

    # Left eye width
    left_eye_width = np.linalg.norm(points[39] - points[36]) + epsilon

    # Right eye width
    right_eye_width = np.linalg.norm(points[45] - points[42]) + epsilon

    # Left eye height
    left_eye_height = (np.linalg.norm(points[38] - points[40]) + np.linalg.norm(points[37] - points[41])) / 2 + epsilon

    # Right eye height
    right_eye_height = (np.linalg.norm(points[44] - points[46]) + np.linalg.norm(points[43] - points[47])) / 2 + epsilon

    # Interocular distance (distance between inner eye corners)
    interocular_distance = np.linalg.norm(points[39] - points[42]) + epsilon

    # Eye-to-eye outer distance (distance between outer eye corners)
    eye_outer_distance = np.linalg.norm(points[36] - points[45]) + epsilon

    # Eyebrow widths
    left_eyebrow_width = np.linalg.norm(points[22] - points[17]) + epsilon
    right_eyebrow_width = np.linalg.norm(points[26] - points[22]) + epsilon

    # Eyebrow heights (from eyebrow to eye)
    left_eyebrow_eye_distance = np.linalg.norm(points[19] - points[37]) + epsilon
    right_eyebrow_eye_distance = np.linalg.norm(points[24] - points[44]) + epsilon

    # Nose to chin distance
    nose_to_chin = np.linalg.norm(points[33] - points[8]) + epsilon

    # Nose to mouth distance
    nose_to_mouth = np.linalg.norm(points[33] - ((points[51] + points[62]) / 2)) + epsilon

    # Calculate face area (approximate as face_width * face_height)
    face_area = face_width * face_height + epsilon

    # Compute centers
    left_eye_center = np.mean(points[36:42], axis=0)
    right_eye_center = np.mean(points[42:48], axis=0)
    mouth_center = np.mean(points[48:68], axis=0)
    eyebrow_midpoint = (points[21] + points[22]) / 2  # Approximate forehead point

    # Cheekbone width (distance between points 1 and 15)
    cheekbone_width = np.linalg.norm(points[1] - points[15]) + epsilon

    # Compute angles
    # Nose angle (angle between nose bridge and tip)
    nose_vector = points[33] - points[27]
    nose_angle = math.atan2(nose_vector[1], nose_vector[0])  # In radians

    # Jaw angles
    left_jaw_vector = points[0] - points[8]
    right_jaw_vector = points[16] - points[8]
    left_jaw_angle = math.atan2(left_jaw_vector[1], left_jaw_vector[0]) + epsilon
    right_jaw_angle = math.atan2(right_jaw_vector[1], right_jaw_vector[0]) + epsilon

    # Compute areas using ConvexHull
    try:
        left_eye_hull = ConvexHull(points[36:42])
        left_eye_area = left_eye_hull.volume + epsilon
    except:
        left_eye_area = epsilon

    try:
        right_eye_hull = ConvexHull(points[42:48])
        right_eye_area = right_eye_hull.volume + epsilon
    except:
        right_eye_area = epsilon

    try:
        mouth_hull = ConvexHull(points[48:60])
        mouth_area = mouth_hull.volume + epsilon
    except:
        mouth_area = epsilon

    # Compute proportions
    # 1. Face width to face height
    proportions.append(face_width / face_height)

    # 2. Nose length to face height
    proportions.append(nose_length / face_height)

    # 3. Nose width to face width
    proportions.append(nose_width / face_width)

    # 4. Mouth width to nose width
    proportions.append(mouth_width / nose_width)

    # 5. Mouth width to face width
    proportions.append(mouth_width / face_width)

    # 6. Eye width to face width (left and right)
    proportions.append(left_eye_width / face_width)
    proportions.append(right_eye_width / face_width)

    # 7. Eye height to eye width (aspect ratio)
    proportions.append(left_eye_height / left_eye_width)
    proportions.append(right_eye_height / right_eye_width)

    # 8. Interocular distance to face width
    proportions.append(interocular_distance / face_width)

    # 9. Eye outer distance to face width
    proportions.append(eye_outer_distance / face_width)

    # 10. Left eye width to right eye width
    proportions.append(left_eye_width / right_eye_width)

    # 11. Left eye height to right eye height
    proportions.append(left_eye_height / right_eye_height)

    # 12. Eyebrow to eye distance (left and right)
    proportions.append(left_eyebrow_eye_distance / face_height)
    proportions.append(right_eyebrow_eye_distance / face_height)

    # 13. Mouth height to mouth width
    proportions.append(mouth_height / mouth_width)

    # 14. Nose width to mouth width
    proportions.append(nose_width / mouth_width)

    # 15. Nose length to nose width
    proportions.append(nose_length / nose_width)

    # 16. Nose to chin distance to face height
    proportions.append(nose_to_chin / face_height)

    # 17. Nose to mouth distance to nose to chin distance
    proportions.append(nose_to_mouth / nose_to_chin)

    # 18. Left eyebrow width to right eyebrow width
    proportions.append(left_eyebrow_width / right_eyebrow_width)

    # 19. Left eyebrow to eye distance to right eyebrow to eye distance
    proportions.append(left_eyebrow_eye_distance / right_eyebrow_eye_distance)

    # 20. Symmetry of facial features (nose to mouth corners)
    nose_to_left_mouth = np.linalg.norm(points[33] - points[48]) + epsilon
    nose_to_right_mouth = np.linalg.norm(points[33] - points[54]) + epsilon
    proportions.append(nose_to_left_mouth / nose_to_right_mouth)

    # 21. Jaw angles ratio (left to right)
    proportions.append(left_jaw_angle / right_jaw_angle)

    # 22. Triangle area between eyes and mouth normalized by face area
    triangle_area = 0.5 * abs(
        (left_eye_center[0]*(right_eye_center[1]-mouth_center[1]) +
         right_eye_center[0]*(mouth_center[1]-left_eye_center[1]) +
         mouth_center[0]*(left_eye_center[1]-right_eye_center[1]))
    ) + epsilon
    proportions.append(triangle_area / face_area)

    # 23. Interocular distance to nose width
    proportions.append(interocular_distance / nose_width)

    # 24. Forehead to chin distance to face height
    forehead_to_chin = np.linalg.norm(points[8] - eyebrow_midpoint) + epsilon
    proportions.append(forehead_to_chin / face_height)

    # 25. Cheekbone width to face width
    proportions.append(cheekbone_width / face_width)

    # 26. Nose angle
    proportions.append(nose_angle)

    # 27. Mouth to chin distance to mouth to nose distance
    mouth_to_chin = np.linalg.norm(mouth_center - points[8]) + epsilon
    mouth_to_nose = np.linalg.norm(mouth_center - points[33]) + epsilon
    proportions.append(mouth_to_chin / mouth_to_nose)

    # 28. Left eye to mouth distance vs. right eye to mouth distance
    left_eye_to_mouth = np.linalg.norm(left_eye_center - mouth_center) + epsilon
    right_eye_to_mouth = np.linalg.norm(right_eye_center - mouth_center) + epsilon
    proportions.append(left_eye_to_mouth / right_eye_to_mouth)

    # 29. Eye area symmetry (left to right)
    proportions.append(left_eye_area / right_eye_area)

    # 30. Mouth area to face area
    proportions.append(mouth_area / face_area)

    # 31. Eye area to face area (left and right)
    proportions.append(left_eye_area / face_area)
    proportions.append(right_eye_area / face_area)

    # 32. Chin angle (relative to horizontal)
    chin_vector = points[8] - ((points[6] + points[10]) / 2)
    chin_angle = math.atan2(chin_vector[1], chin_vector[0]) + epsilon
    proportions.append(chin_angle)

    # 33. Distance between eyes to mouth center
    eyes_to_mouth = np.linalg.norm(((left_eye_center + right_eye_center) / 2) - mouth_center) + epsilon
    proportions.append(eyes_to_mouth / face_height)

    # 34. Eye eccentricity (shape descriptor)
    left_eye_eccentricity = left_eye_height / left_eye_width
    right_eye_eccentricity = right_eye_height / right_eye_width
    proportions.append(left_eye_eccentricity)
    proportions.append(right_eye_eccentricity)

    # 35. Distance from center of mouth to chin
    mouth_center_to_chin = np.linalg.norm(mouth_center - points[8]) + epsilon
    proportions.append(mouth_center_to_chin / face_height)

    # 36. Facial symmetry based on distances from midline
    mid_x = (points[0][0] + points[16][0]) / 2
    left_face = np.mean(points[:17, 0] - mid_x) + epsilon
    right_face = np.mean(mid_x - points[:17, 0]) + epsilon
    proportions.append(left_face / right_face)

    # 37. Ratio of upper lip height to lower lip height
    upper_lip_height = np.linalg.norm(points[51] - points[62]) + epsilon
    lower_lip_height = np.linalg.norm(points[66] - points[57]) + epsilon
    proportions.append(upper_lip_height / lower_lip_height)

    # 38. Proportion of nose area to face area (approximated)
    try:
        nose_points = points[27:36]
        nose_hull = ConvexHull(nose_points)
        nose_area = nose_hull.volume + epsilon
    except:
        nose_area = epsilon
    proportions.append(nose_area / face_area)

    # 39. Mouth to nose distance to eye to nose distance
    eye_to_nose = np.linalg.norm(((left_eye_center + right_eye_center) / 2) - points[33]) + epsilon
    proportions.append(nose_to_mouth / eye_to_nose)

    # 40. Proportion of eye spacing to face width
    eye_spacing = np.linalg.norm(points[39] - points[42]) + epsilon
    proportions.append(eye_spacing / face_width)

    return proportions
