import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define directories
input_dir = "./photos/unknown"
output_dir = "output111"
sketches_dir = os.path.join(output_dir, "sketches34")
features_base_dir = os.path.join(output_dir, "features")
feature_dirs = {
    "eyes": os.path.join(features_base_dir, "eyes"),
    "lips": os.path.join(features_base_dir, "lips"),
    "nose": os.path.join(features_base_dir, "nose"),
    "eyebrows": os.path.join(features_base_dir, "eyebrows"),
    "face": os.path.join(features_base_dir, "face"),
    "hair": os.path.join(features_base_dir, "hair")
}

# Create directories
os.makedirs(sketches_dir, exist_ok=True)
for dir_path in feature_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Function to convert image to sketch
def image_to_sketch(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to reduce noise
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection with Canny
    edges = cv2.Canny(blurred, 30, 100)
    # Invert edges for sketch effect
    sketch = cv2.bitwise_not(edges)
    return sketch

# Function to create a mask for a specific facial feature
def create_feature_mask(image_shape, landmarks, indices):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = [(int(landmarks[i].x * image_shape[1]), int(landmarks[i].y * image_shape[0])) for i in indices]
    if points:
        hull = cv2.convexHull(np.array(points, dtype=np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    return mask

# Function to create a mask for the hair region
def create_hair_mask(image_shape, landmarks):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    forehead = (int(landmarks[10].x * image_shape[1]), int(landmarks[10].y * image_shape[0]))  # Forehead point
    top_left = (int(landmarks[103].x * image_shape[1]), int(landmarks[103].y * image_shape[0] * 0.8))
    top_right = (int(landmarks[332].x * image_shape[1]), int(landmarks[332].y * image_shape[0] * 0.8))
    points = [top_left, forehead, top_right]
    points_array = np.array(points, dtype=np.int32).reshape(-1, 1, 2)  # Ensure shape (N, 1, 2)
    hull = cv2.convexHull(points_array)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

# Define facial feature indices (MediaPipe Face Mesh)
feature_indices = {
    "left_eye": ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], "eyes"),
    "right_eye": ([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], "eyes"),
    "lips": ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95], "lips"),
    "nose": ([1, 2, 98, 327, 331, 294, 278, 168], "nose"),
    "left_eyebrow": ([46, 53, 52, 65, 55, 70, 63, 105, 66, 107], "eyebrows"),
    "right_eyebrow": ([285, 295, 282, 283, 276, 300, 293, 334, 296, 336], "eyebrows"),
    "face": ([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], "face")
}

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {filename}")
            continue

        # Generate sketch
        sketch = image_to_sketch(image)
        sketch_path = os.path.join(sketches_dir, f"sketch_{filename}")
        cv2.imwrite(sketch_path, sketch)

        # Convert sketch to RGB for MediaPipe
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        results = face_mesh.process(sketch_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                # Create a transparent base image for features
                transparent_base = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

                # Process each feature
                for feature_name, (indices, feature_dir) in feature_indices.items():
                    mask = create_feature_mask(image.shape, landmarks, indices)
                    # Apply mask to sketch
                    feature_image = transparent_base.copy()
                    feature_image[:, :, 3] = mask  # Alpha channel
                    feature_image[:, :, :3] = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)  # Sketch content
                    feature_image[mask == 0] = [0, 0, 0, 0]  # Make non-feature areas transparent
                    # Save feature
                    feature_filename = f"{filename.split('.')[0]}_{feature_name}.png"
                    feature_path = os.path.join(feature_dirs[feature_dir], feature_filename)
                    Image.fromarray(feature_image).save(feature_path)

                # Process hair
                hair_mask = create_hair_mask(image.shape, landmarks)
                hair_image = transparent_base.copy()
                hair_image[:, :, 3] = hair_mask
                hair_image[:, :, :3] = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
                hair_image[hair_mask == 0] = [0, 0, 0, 0]
                hair_filename = f"{filename.split('.')[0]}_hair.png"
                hair_path = os.path.join(feature_dirs["hair"], hair_filename)
                Image.fromarray(hair_image).save(hair_path)

        print(f"Processed {filename}")

# Release MediaPipe resources
face_mesh.close()