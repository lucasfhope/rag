import fitz  # PyMuPDF
import io
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import hashlib
import matplotlib.pyplot as plt

def image_to_hash(image):
    """
    Convert image to a hash using its pixel data.
    """
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()

def extract_text_and_images_from_pdf(pdf_path, min_width=200, min_height=200):
    doc = fitz.open(pdf_path)
    text = ""
    images = []
    image_hashes = set()
    text_data = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        text += page_text
        text_data.append((page_num, page_text))

        # Extract images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Filter out small images (e.g., logos and banners)
            if image.width >= min_width and image.height >= min_height:
                # Generate hash for the image
                img_hash = image_to_hash(image)

                # Check if the image is already added
                if img_hash not in image_hashes:
                    images.append((image, page_num))
                    image_hashes.add(img_hash)

    return text, images, text_data

pdf_path = '11-LLM-Prompting-Lecture.pdf'
text, images, text_data = extract_text_and_images_from_pdf(pdf_path)

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

# Load a pre-trained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_image_features(image):
    image = preprocess(image).unsqueeze(0)  # Preprocess and add a batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()  # Remove the batch dimension and convert to numpy array

# Extract features for all images
image_features = [extract_image_features(image) for image, _ in images]

# Create a FAISS index
d = image_features[0].shape[0]  # Dimension of the feature vectors
index = faiss.IndexFlatL2(d)  # Use L2 distance

# Add the image features to the index
index.add(np.array(image_features))

# Save the index to disk (optional)
faiss.write_index(index, "image_features.index")

# Function to extract text from a specific page
def extract_text_from_page(doc, page_num):
    page = doc.load_page(page_num)
    return page.get_text()

# Example of querying similar images
for image, page_num in images:
    features = extract_image_features(image).reshape(1, -1)

    # Perform the search
    k = 5  # Number of nearest neighbors
    distances, indices = index.search(features, k)

    # Retrieve the top-k similar images and their page numbers
    similar_images = [images[i] for i in indices[0]]

    # Print the indices of the similar images and their associated text
    print(f"Original Image on Page {page_num + 1}:")
    for i, (img, img_page_num) in enumerate(similar_images):
        similar_text = extract_text_from_page(fitz.open(pdf_path), img_page_num)
        print(f"  Similar Image {i + 1} on Page {img_page_num + 1}:")
        print(f"    Text: {similar_text[:500]}")  # Print the first 500 characters of the associated text

# Collect text associated with images for RAG usage
image_text_pairs = [(img, extract_text_from_page(fitz.open(pdf_path), page_num)) for img, page_num in images]

# Example usage of image_text_pairs in a RAG model
for img, img_text in image_text_pairs:
    # Show the image and its associated text
    plt.imshow(img)
    plt.title(img_text[:100])  # Display the first 100 characters of the associated text
    plt.axis('off')
    plt.show()
    # Here, you can pass img and img_text to your RAG model as context


