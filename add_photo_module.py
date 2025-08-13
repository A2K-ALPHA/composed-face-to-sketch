import numpy as np
import tensorflow as tf
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivymd.uix.button import MDRaisedButton
from kivy.uix.filechooser import FileChooserIconView
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import os

import shutil

class AddPhotoPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(title="Select a Photo", size_hint=(0.9, 0.9), **kwargs)
        layout = BoxLayout(orientation='vertical')
        self.chooser = FileChooserIconView(filters=["*.jpg", "*.jpeg", "*.png"])
        add_button = MDRaisedButton(text="Add Photo", on_release=self.add_photo)
        layout.add_widget(self.chooser)
        layout.add_widget(add_button)
        self.content = layout

        # Load model with Dense(128) to match saved embeddings
        print("🔄 Loading face embedding model...")
        base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        x = tf.keras.layers.Dense(128)(base_model.output)
        output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        self.model = Model(inputs=base_model.input, outputs=output)
        print("✅ Model loaded with output size:", self.model.output_shape)

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def add_photo(self, instance):
        selected = self.chooser.selection
        if selected:
            img_path = selected[0]
            save_folder = "./output_photos"
            os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
            filename = os.path.basename(img_path)    # Get the file name
            saved_path = os.path.join(save_folder, filename)
            shutil.copy(img_path, saved_path)
            print(f"📸 Selected image: {img_path}")
            img_data = self.preprocess_image(img_path)
            embedding = self.model.predict(img_data)[0]
            print("✅ Generated embedding with shape:", embedding.shape)

            try:
                embeddings = np.load("face_embeddings.npy", allow_pickle=True)
                paths = np.load("face_paths.npy", allow_pickle=True)
                labels = np.load("face_labels.npy", allow_pickle=True)
                print("📂 Loaded existing database files.")
            except FileNotFoundError:
                print("⚠️ No existing database found. Creating new ones.")
                embeddings = np.empty((0, 128))
                paths = np.array([])
                labels = np.array([])

            label = os.path.splitext(os.path.basename(img_path))[0]

            embeddings = np.vstack([embeddings, embedding])
            paths = np.append(paths, img_path)
            labels = np.append(labels, label)

            np.save("face_embeddings.npy", embeddings)
            np.save("face_paths.npy", paths)
            np.save("face_labels.npy", labels)
            print(f"✅ Photo added successfully: {label}")

            self.dismiss()
        else:
            print("❌ No photo selected.")

def open_add_photo_popup():
    popup = AddPhotoPopup()
    popup.open()
