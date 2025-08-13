import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

class PhotoDatabaseManager:
    def __init__(self):
        # Load the trained model
        self.model = load_model("model_sketch_best_vgg_mask.h5", compile=False)
        
        # Load existing embeddings and labels
        self.embeddings = np.load("face_embeddings.npy") if os.path.exists("face_embeddings.npy") else np.empty((0, 128))
        self.labels = np.load("face_labels.npy") if os.path.exists("face_labels.npy") else np.array([])
        self.paths = np.load("face_paths.npy") if os.path.exists("face_paths.npy") else np.array([])

    def preprocess(self, img_path):
        """Preprocess image for model prediction."""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256)).astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    def add_photo(self, new_photo_path):
        """Add a new photo to the database."""
        # Generate label from the file name (without extension)
        new_label = os.path.splitext(os.path.basename(new_photo_path))[0]
        
        # Process and embed the new photo
        new_img = self.preprocess(new_photo_path)
        new_embedding = self.model.predict(new_img)[0]
        
        # Append new data to existing arrays
        self.embeddings = np.vstack([self.embeddings, new_embedding])
        self.labels = np.append(self.labels, new_label)
        self.paths = np.append(self.paths, new_photo_path)
        
        # Save back to .npy files
        np.save("face_embeddings.npy", self.embeddings)
        np.save("face_labels.npy", self.labels)
        np.save("face_paths.npy", self.paths)
        
        print(f"✅ Added new face: {new_label}")

    def get_embeddings(self):
        """Return current database embeddings."""
        return self.embeddings

    def get_labels(self):
        """Return current database labels."""
        return self.labels

    def get_paths(self):
        """Return current database paths."""
        return self.paths
