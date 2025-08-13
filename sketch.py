import cv2
from kivy.uix.scatter import Scatter
from kivy.uix.image import Image
from kivy.graphics import Fbo, ClearColor, ClearBuffers, Scale, Translate
from kivy.core.image import Image as CoreImage
from kivy.properties import BooleanProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.label import Label
from kivymd.uix.button import MDRaisedButton
from kivy.uix.screenmanager import Screen
from kivy.uix.stencilview import StencilView
from kivymd.uix.relativelayout import MDRelativeLayout
from gan_method import generate_image_with_gan
import os
from PIL import Image as PILImage 
import glob
from bharat1 import align_face
from compose import compose_face_from_parts
class DraggableImage(Scatter):
    def __init__(self, source, **kwargs):
        super().__init__(do_rotation=False, do_scale=True, do_translation=True, **kwargs)
        self.size_hint = (None, None)
        self.size = (100, 100)
        self.image = Image(source=source, size=self.size)
        self.add_widget(self.image)
        self._dragging = False

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):  # If the touch is within the bounds of the image
            self._dragging = True
            self.image.opacity = 0.7  # Optionally, dim the image to show it's being dragged
            return super().on_touch_down(touch)  # Allow the touch to pass to Scatter for translation
        return False  # Prevent others from being dragged

    def on_touch_move(self, touch):
        if self._dragging:  # Only move the image if it's being dragged
            return super().on_touch_move(touch)  # Let Scatter handle the movement of the image
        return False

    def on_touch_up(self, touch):
        if self._dragging:
            self._dragging = False
            self.image.opacity = 1  # Restore the opacity once the dragging is done
            return super().on_touch_up(touch)
        return False


class FixedImage(Scatter):
    def __init__(self, category, source, size, pos, **kwargs):
        super().__init__(do_rotation=False, do_scale=True, do_translation=True, **kwargs)
        self.size_hint = (None, None)
        self.size = size
        self.image = Image(source=source, size=self.size)
        self.add_widget(self.image)
        self.pos = pos

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):  # Only allow touch events for the image that's being clicked
            return super().on_touch_down(touch)  # Allow the image to react to touch
        return False


class SketchScreen(Screen):
    show_images = BooleanProperty(False)
    active_images = {}
    img_source={}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_paths = self.load_images()

    def on_kv_post(self, base_widget):
        self.setup_canvas_layers()

    def setup_canvas_layers(self):
        self.stencil = StencilView(size_hint=(None, None))
        self.ids.canvas_area.bind(size=self.update_stencil, pos=self.update_stencil)
        self.stencil.size = self.ids.canvas_area.size
        self.stencil.pos = self.ids.canvas_area.pos

        # Background
        self.background_layer = MDRelativeLayout(size_hint=(None, None))
        self.background_layer.size = self.ids.canvas_area.size
        self.background_layer.pos = self.ids.canvas_area.pos

        # Foreground
        self.foreground_layer = MDRelativeLayout(size_hint=(None, None))
        self.foreground_layer.size = self.ids.canvas_area.size
        self.foreground_layer.pos = self.ids.canvas_area.pos

        self.stencil.add_widget(self.background_layer)
        self.stencil.add_widget(self.foreground_layer)

        self.ids.canvas_area.clear_widgets()
        self.ids.canvas_area.add_widget(self.stencil)
        self.ids.canvas_area.bind(size=self.update_layers, pos=self.update_layers)

    def update_stencil(self, *args):
        self.stencil.size = self.ids.canvas_area.size
        self.stencil.pos = self.ids.canvas_area.pos

    def update_layers(self, *args):
        self.background_layer.size = self.ids.canvas_area.size
        self.background_layer.pos = self.ids.canvas_area.pos
        self.foreground_layer.size = self.ids.canvas_area.size
        self.foreground_layer.pos = self.ids.canvas_area.pos

    def save_canvas(self,name):
        width, height = map(int, self.ids.canvas_area.size)
        fbo = Fbo(size=(width, height), with_stencilbuffer=True)
        """" head=self.active_images["head"]
        eyes=self.active_images["eyes"]
        eyebrows=self.active_images["eyebrows"]
        nose=self.active_images["nose"]
        mouth=self.active_images["mouth"]
        hair=self.active_images["hair"]
        compose_face_from_parts(
            image_id="001",
            head_img=head,
            eyes_img=eyes,
            eyebrows_img=eyebrows,
            nose_img=nose,
            mouth_img=mouth,
            hair_img=hair,
            output_path="output_faces/001_composed.png"
        )"""
        head = PILImage.open(self.img_source["head"]).convert("RGBA")
        eyes = PILImage.open(self.img_source["eyes"]).convert("RGBA")
        eyebrows = PILImage.open(self.img_source["eyebrows"]).convert("RGBA")
        nose = PILImage.open(self.img_source["nose"]).convert("RGBA")
        mouth = PILImage.open(self.img_source["mouth"]).convert("RGBA")
        hair = PILImage.open(self.img_source["hair"]).convert("RGBA")
        compose_face_from_parts(
            image_id="001",
            head_img=head,
            eyes_img=eyes,
            eyebrows_img=eyebrows,
            nose_img=nose,
            mouth_img=mouth,
            hair_img=hair,
            output_path="output_faces/001_composed.png")
        with fbo:
            # Clear with white color (1, 1, 1, 1) - white background
            ClearColor(1, 1, 1, 1)
            ClearBuffers()  # Clear color and depth buffers
            Scale(1, -1, 1)  # Flip vertically to save image correctly
            Translate(-self.stencil.x, -self.stencil.y - height, 0)

            # Draw the background layer and foreground layer
            fbo.add(self.background_layer.canvas)
            fbo.add(self.foreground_layer.canvas)

        # Save the final image
        fbo.draw()
        m="./elements/f{name}.png"
        CoreImage(fbo.texture).save(m)
        img=align_face(m)
        if img is not None:
            cv2.imwrite(m, img)
            print(f"Saved aligned image to: {m}")
        else:
            print("No face detected in the image.")

        print("Canvas saved to './elements/canvas_image.png' with a white background.")
        generate_image_with_gan('vgg_15.h5',m,'./saved_sketches/f{name}.png')
        

    def load_images(self):
        images = {}
        for entry in os.scandir("./elements1"):
            if entry.is_dir():
                category = entry.name
                clustered_path = os.path.join(entry.path, "clustered")
                images[category] = {}

                if os.path.exists(clustered_path):
                    for cluster_dir in sorted(os.listdir(clustered_path)):
                        full_cluster_dir = os.path.join(clustered_path, cluster_dir)
                        if os.path.isdir(full_cluster_dir):
                            cluster_name = cluster_dir
                            images[category][cluster_name] = glob.glob(os.path.join(full_cluster_dir, "*.png")) + glob.glob(os.path.join(full_cluster_dir, "*.jpg"))
                            print(f"Loaded images for {category}/{cluster_name}: {images[category][cluster_name]}")  # Debug line
                else:
                    images[category]["default"] = glob.glob(os.path.join(entry.path, "*.png")) + glob.glob(os.path.join(entry.path, "*.jpg"))
                    print(f"Loaded default images for {category}: {images[category]['default']}")  # Debug line
        return images

    def toggle_images(self, category):
        self.show_images = not self.show_images
        self.populate_images(category)

    def populate_images(self, category):
        self.ids.image_row.clear_widgets()

        if self.show_images and category in self.image_paths:
            for cluster_name, img_list in self.image_paths[category].items():
                cluster_button = MDRaisedButton(
                    text=cluster_name,
                    size_hint=(None, None),
                    size=(120, 40),
                    on_release=lambda instance, cluster=cluster_name: self.load_cluster_images(cluster, category)
                )
                self.ids.image_row.add_widget(cluster_button)

    def load_cluster_images(self, cluster_name, category):
        self.ids.image_row.clear_widgets()

        cluster_images = self.image_paths[category].get(cluster_name, [])
        if cluster_images:
            for img_path in cluster_images:
                img_widget = Image(source=img_path, size_hint=(None, None), size=(100, 100))
                img_widget.bind(on_touch_down=lambda instance, touch, path=img_path, cat=category:
                                self.on_image_selected(instance, touch, path, cat))
                self.ids.image_row.add_widget(img_widget)
        else:
            no_images_label = Label(text=f"No images in {cluster_name}", size_hint=(None, None), size=(120, 40))
            self.ids.image_row.add_widget(no_images_label)

    def on_image_selected(self, instance, touch, path, category):
        if instance.collide_point(*touch.pos):  # Ensure the touch is on the image
            self.add_fixed_image(path, category)

    def add_fixed_image(self, img_path, category):
        if category in self.active_images:
            old = self.active_images.pop(category)
            if old.parent:
                old.parent.remove_widget(old)

        size_map = {
            "head": (300, 400),
            "eyes": (60, 30),
            "eyebrows": (70, 20),
            "nose": (50, 80),
            "mouth": (60, 30),
            "ears": (40, 70),
            "mustache": (70, 30),
            "hair": (300, 120)
        }

        pos_map = {
            "head": (0, 0),
            "eyes": (145, 260),
            "eyebrows": (145, 290),
            "nose": (140, 200),
            "mouth": (140, 150),
            "ears": (140, 240),
            "mustache": (140, 180),
            "hair": (0, 200)
        }

        size = size_map.get(category, (100, 100))
        pos = pos_map.get(category, (150, 150))
        fixed_img = FixedImage(category=category, source=img_path, size=size, pos=pos)
        self.img_source[category]=img_path
        # Add hair and head to the background layer
        if category == "head" or category == "hair":
            self.background_layer.add_widget(fixed_img)
        # Add other body parts (like eyes, nose, etc.) to the foreground layer
        else:
            self.foreground_layer.add_widget(fixed_img)

        self.active_images[category] = fixed_img
