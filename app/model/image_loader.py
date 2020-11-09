from PIL import Image
import numpy as np

class ImageLoader:
    
    @classmethod
    def from_file(cls, img_path):
        img = np.asarray(Image.open(img_path).convert("RGB").resize((160, 160)), np.uint8)

        return img
    
    @classmethod
    def from_view(cls):
        """
        Load input image from interface (View)
        """
        pass
    