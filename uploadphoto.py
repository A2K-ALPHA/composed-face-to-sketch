import cv2
import numpy as np
import os

from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import DenseNet121
import tensorflow as tf
from dlibe import get_top_procrustes_matches
from compose import compose_face_from_parts
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivymd.uix.button import MDRaisedButton
from landmark_matcher import get_top_landmark_matches, extract_landmarks
from kivy.uix.image import Image as KivyImage
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from testgan import run_inference
from bharat1 import align_face
from kivy.graphics.texture import Texture # ⬅️ imported functions
def compare_embeddings(img1_path, img2_path):
    from deepface import DeepFace
    from sklearn.metrics.pairwise import cosine_similarity

    emb1 = DeepFace.represent(img_path=img1_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
    emb2 = DeepFace.represent(img_path=img2_path, model_name="ArcFace", enforce_detection=True)[0]["embedding"]

    return cosine_similarity([emb1], [emb2])[0][0]

class UploadSketchPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(title="Upload a Sketch", size_hint=(0.9, 0.9), **kwargs)
        layout = BoxLayout(orientation='vertical')
        self.chooser = FileChooserIconView(filters=["*.jpg", "*.jpeg", "*.png"])
        match_button = MDRaisedButton(text="Generate & Match", on_release=self.process_and_match)
        layout.add_widget(self.chooser)
        layout.add_widget(match_button)
        self.content = layout

        # Load GAN model
        self.gan_model = load_model(r"C:\Users\Bharat\Downloads\CelebAMask-HQa\CelebAMask-HQ\ui\model_chinese3\model_gen_chhhh_best_1.h5")
        print("✅ GAN model loaded")

        # Load DenseNet121 for embeddings
        base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
        x = tf.keras.layers.Dense(128)(base_model.output)
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        self.embedding_model = Model(inputs=base_model.input, outputs=x)
        print("✅ siamese DenseNet121 embedding model loaded")

    def preprocess_sketch(self, sketch_path):
        img=align_face(sketch_path)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = (img_resized / 127.5) - 1
        return np.expand_dims(img_normalized, axis=0)

    def extract_embeddings(self, img):
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        embeddings = self.embedding_model.predict(img_expanded)
        return embeddings

    def procrustes_similarity(self, landmarks1, landmarks2):
        if landmarks1.shape != landmarks2.shape:
            return float("inf")
        mtx1, mtx2, disparity = procrustes(landmarks1, landmarks2)
        return disparity

    def process_and_match(self, instance):
        selected = self.chooser.selection
        if not selected:
            print("⚠️ No sketch selected.")
            return

        sketch_path = selected[0]
        print(f"📥 Sketch selected: {sketch_path}")
        
        # Step 1: Generate photo from sketch
        sketch = self.preprocess_sketch(sketch_path)
        generated_photo = self.gan_model.predict(sketch)[0]
        generated_photo = ((generated_photo + 1) * 127.5).astype(np.uint8)
        generated_photo=run_inference(sketch_path)
        

        # Step 2: Get top landmark matches
        sketch_img = cv2.imread(sketch_path)
        top_landmark_matches = get_top_landmark_matches(sketch_img, folder_path="./crisp_images1", top_k=10)

        print("✅ Top 10 dlib landmark matches retrieved")
        for idx, (path, score) in enumerate(top_landmark_matches, 1):
            print(f"{idx:2d}. {os.path.basename(path)} - Disparity: {score:.4f}")

        # Step 3: DenseNet + Procrustes on top matches
        from deepface import DeepFace

        # Save generated image temporarily
        """temp_path = "./elements/generated_output.jpg"
        photo=self.preprocess_sketch(temp_path)
        generated_photo=self.gan_model(photo,training=False)[0]
        output_image = ((generated_photo + 1) * 127.5).numpy().astype(np.uint8)"""


        # Get DeepFace embedding for GAN output
        gen_embedding = DeepFace.represent(generated_photo, model_name="ArcFace", enforce_detection=False)[0]["embedding"]

        final_scores = []
        for match_path, _ in top_landmark_matches:
            try:
                real_embedding = DeepFace.represent(img_path=match_path, model_name="ArcFace", enforce_detection=True)[0]["embedding"]
                cosine_sim = cosine_similarity([gen_embedding], [real_embedding])[0][0]
                final_scores.append((match_path, cosine_sim))
            except Exception as e:
                print(f"❌ Failed to process {match_path}: {e}")
                continue

        # Sort by descending similarity (highest match first)
        final_scores.sort(key=lambda x: -x[1])
        top_10_final = [(p, 1 - sim, 0.0, sim) for p, sim in final_scores[:10]]  # match existing format

        print("\n✅ Final top matches:")
        for i, (path, comb, proc, cos) in enumerate(top_10_final, 1):
            print(f"{i}. {os.path.basename(path)} | Combined: {comb:.4f}, Cosine: {cos:.4f}")

        self.show_result_popup(sketch_path, generated_photo, top_10_final)



    def show_result_popup(self, sketch_path, generated_photo, top_10_matches):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # === Top row: Sketch and Generated photo ===
        top_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=160)

        # Sketch image
        sketch_img = cv2.imread(sketch_path)
        sketch_widget = self.cv2_to_kivy_image(sketch_img)
        top_row.add_widget(sketch_widget)

        # Generated photo
        gen_widget = self.cv2_to_kivy_image(generated_photo)
        top_row.add_widget(gen_widget)

        layout.add_widget(Label(text="Sketch (left) and GAN Output (right)", size_hint_y=None, height=30))
        layout.add_widget(top_row)

        # === Scrollable section for top 10 matches ===
        scroll_view = ScrollView(size_hint=(1, 1))
        match_grid = GridLayout(cols=1, spacing=10, size_hint_y=None)
        match_grid.bind(minimum_height=match_grid.setter('height'))

        for idx, (match_path, score, pro_score, cos_score) in enumerate(top_10_matches, 1):
            match_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=150)

            # Matched image
            match_img = cv2.imread(match_path)
            match_widget = self.cv2_to_kivy_image(match_img)
            match_box.add_widget(match_widget)

            # Info label
            info_text = (
                f"[b]Match {idx}[/b]\n"
                f"{os.path.basename(match_path)}\n"
                f"Combined Score: {score:.4f}\n"
                f"Procrustes: {pro_score:.4f}, Cosine: {cos_score:.4f}"
            )
            info_label = Label(text=info_text, markup=True, halign='left', valign='middle', size_hint_x=2)
            info_label.bind(size=info_label.setter('text_size'))
            match_box.add_widget(info_label)

            match_grid.add_widget(match_box)

        scroll_view.add_widget(match_grid)
        layout.add_widget(scroll_view)

        popup = Popup(title="Top 10 Match Results", content=layout, size_hint=(0.95, 0.95))
        popup.open()


    def cv2_to_kivy_image(self, img):
        """Convert a BGR OpenCV image to a Kivy Image widget."""
        if img is None:
            return Label(text="Image not found")
        print(f"🖼️ Image:| Shape: {img.shape}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (256, 256))
        buf = img.tobytes()

        texture = Texture.create(size=(256, 256))
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        return KivyImage(texture=texture, size_hint_x=1)



    def open_upload_sketch_popup():
        popup = UploadSketchPopup()
        popup.open()
