import cv2
import dlib
import os
import imutils

# Load dlib's face detector and face landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure you have this file

def align_face(image_path, output_path, padding_top=0.2, padding_bottom=0.2, padding_left=0.2, padding_right=0.2, size=150):
    # Load the image
    img = cv2.imread(image_path)
    img = imutils.resize(img, width=500)  # Resize the image to a manageable size
    
    # Detect faces in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)
        
        # Get the bounding box of the face (left, top, right, bottom)
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
        # Calculate padding in each direction
        padding_x_left = int(w * padding_left)  # Left padding
        padding_x_right = int(w * padding_right)  # Right padding
        padding_y_top = int(h * padding_top)  # Top padding
        padding_y_bottom = int(h * padding_bottom)  # Bottom padding
        
        # Extend the bounding box in all four directions (left, right, top, bottom)
        x_padded = max(0, x - padding_x_left)
        y_padded = max(0, y - padding_y_top)
        w_padded = min(img.shape[1], x + w + padding_x_right) - x_padded
        h_padded = min(img.shape[0], y + h + padding_y_bottom) - y_padded
        
        # Crop the face with padding
        cropped_face_with_padding = img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        
        # Resize the cropped face with padding to a standard size
        aligned_face = cv2.resize(cropped_face_with_padding, (size, size))
        
        # Save the aligned face to the output path
        cv2.imwrite(output_path, aligned_face)
        print(f"Aligned face saved to {output_path}")

def process_batch(input_folder, output_folder, padding_top=0.2, padding_bottom=0.2, padding_left=0.2, padding_right=0.2, size=150):
    # Get all image paths in the input folder
    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    # Process each image in the folder
    for image_path in image_paths:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        if '_light_sketch' in filename:
            filename= filename.replace('_light_sketch', '')
        
        # Construct the output path
        output_path = os.path.join(output_folder, f"{filename}.jpg")
        
        # Align the face and save it
        align_face(image_path, output_path, padding_top, padding_bottom, padding_left, padding_right, size)

# Example usage
input_folder = r"C:\Users\Bharat\Downloads\CelebAMask-HQa\CelebAMask-HQ\ui\output_faces"  # Folder containing your input images
output_folder = r"C:\Users\Bharat\Downloads\CelebAMask-HQa\CelebAMask-HQ\ui\saved_sketches1"  # Folder to save aligned images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process the batch of images
process_batch(input_folder, output_folder, padding_top=0.7, padding_bottom=0.2, padding_left=0.2, padding_right=0.2, size=150)
