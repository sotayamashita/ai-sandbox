from PIL import Image


def show_image(image_path: str, max_size: int = 800):
    image = Image.open(image_path)
    image.thumbnail((max_size, max_size))
    image.show()
