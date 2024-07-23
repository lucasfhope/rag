import fitz  # PyMuPDF
import io
from PIL import Image

def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    images = []

    for page in doc:
        text += page.get_text()

        # Extract images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    return text, images

pdf_path = '11-LLM-Prompting-Lecture.pdf'
text, images = extract_text_and_images_from_pdf(pdf_path)

print("Extracted Text:", text[:500])  # Print first 500 characters of the text
print("Number of Images Extracted:", len(images))

import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

preprocessed_text = preprocess_text(text)
print("Preprocessed Text:", preprocessed_text[:500])





from PIL import Image

# Open an image file
image_path = 'path/to/your/image.jpg'  # Replace with your image file path
image = Image.open(image_path)

# Get image size
width, height = image.size
print(f'Width: {width}, Height: {height}')

# Get image mode (e.g., 'RGB', 'RGBA', 'L', etc.)
mode = image.mode
print(f'Mode: {mode}')

# Get pixel data
pixel_data = list(image.getdata())
print(f'Pixel Data: {pixel_data[:10]}')  # Print the first 10



# Load image data into a pixel access object
pixels = image.load()

# Get the value of a specific pixel (e.g., at position (x, y))
x, y = 0, 0  # Coordinates of the pixel
pixel_value = pixels[x, y]
print(f'Pixel value at ({x}, {y}): {pixel_value}')

import numpy as np

# Convert image to numpy array
image_array = np.array(image)
print(f'Image Array:\n{image_array}')