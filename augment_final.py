import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from numpy import asarray
import csv
from scipy.ndimage import gaussian_filter, map_coordinates
import pickle

#Elastic deformation helper function
def elastic_deform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).flatten(), (x + dx).flatten()

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(shape)

# Function that applies all the augmentations
def augment_image(img, degrees=[d for d in range(0, 360, 10)], brightness_factors=[0.5, 1.5], alpha=20, sigma=5):
    augmented_images = []

    # Rotation
    for angle in degrees:
        rotated = img.rotate(angle)
        augmented_images.append(asarray(rotated))

    # Brightness Adjustment
    enhancer = ImageEnhance.Brightness(img)
    for factor in brightness_factors:
        brightened = enhancer.enhance(factor)
        augmented_images.append(asarray(brightened))

    # Mirroring (Horizontal and Vertical Flips)
    mirrored_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip
    mirrored_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)   # Vertical flip
    augmented_images.append(asarray(mirrored_horizontal))
    augmented_images.append(asarray(mirrored_vertical))

    # Elastic Deformation (if needed)
    img_array = asarray(img.convert('L'))  # Grayscale conversion for deformation
    for _ in range(3):  # Generate 3 elastic variations
        elastic_img = elastic_deform(img_array, alpha, sigma)
        elastic_img = Image.fromarray(elastic_img).convert('RGBA')  # Convert back to RGBA
        elastic_img = elastic_img.resize(img.size)  # Ensure consistent size
        augmented_images.append(asarray(elastic_img))

    return augmented_images

#Function that applies all the augmentations
#def augment_image(img, degrees=[d for d in range(0, 360, 10)], brightness_factors=[0.5, 1.5], shifts=[-10, 10], alpha=20, sigma=5):
    augmented_images = []
    img_array= np.asarray(img)
    
    for angle in degrees:
        rotated = img.rotate(angle)

        #remove the unwanted corner areas by doing the following
        processed_img = Image.alpha_composite(img,rotated)
        augmented_images.append(asarray(processed_img))

    #Changing the brightness to cater to different brightnesses
    enhancer = ImageEnhance.Brightness(img)
    for factor in brightness_factors:
        brightened = enhancer.enhance(factor)
        augmented_images.append(asarray(brightened))
    """
    # Positional Shifts
    for shift_x in shifts:
        for shift_y in shifts:
            shifted = Image.new("L", img.size, 255)
            shifted.paste(img.convert('L'), (shift_x, shift_y))
            augmented_images.append(asarray(shifted))
    
    # Elastic Deformation
    img_array = asarray(img.convert('L'))
    for _ in range(3):  # Generate 3 elastic variations
        elastic_img = elastic_deform(img_array, alpha, sigma)
        augmented_images.append(elastic_img)
    """
    return augmented_images

"""
image_size = (64,64)
img = Image.open("ML/project/train/ई/0.jpg").resize(image_size).convert('RGBA')
print(img.size)
#img.show()

new_images = augment_image(img)
os.chdir("ML/project/augment")
print(len(new_images))
counter = 1
for a in new_images:
    data = Image.fromarray(a) 
    data.save(f'{counter}.png')
    counter += 1
    """

# Main preprocessing function
def preprocess_images(folder_path, apply_augmentation=True):
    x, y = [], []
    image_size = (64, 64)
    #print(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            #print(filename)
            if filename.endswith('.jpg') or filename.endswith('.png'):
                #print('hi')
                file_path = os.path.join(root, filename)
                img = Image.open(file_path).resize(image_size).convert('RGBA')  # Resize and grayscale

                numpydata = asarray(img)
                numpydata = numpydata.reshape(-1)  # Flatten
                #to write out the class, retain the directory name
                dirname = root.split(os.path.sep)[-1]
                x.append(numpydata)
                y.append(dirname)
                
                if apply_augmentation:
                    augmented_images = augment_image(img)
                    for augmented in augmented_images:
                        #print(augmented.shape)
                        x.append(augmented.flatten())
                        y.append(root.split(os.path.sep)[-1])

    return np.array(x), np.array(y)
"""
x, y = preprocess_images('ML/project/train')
print(len(x),len(y))
print(x[0].shape)
"""
def load_validation_data(val_folder_path, label_file):
    x_val, y_val = [], []

    # Load label dictionary
    with open(label_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        labels = {row[0].strip(): row[1].strip() for row in reader}

    for filename in os.listdir(val_folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(val_folder_path, filename)
            img = Image.open(file_path).resize((64, 64)).convert('RGBA')

            x_val.append(asarray(img).flatten())
            y_val.append(labels.get(filename, None))

    return np.array(x_val), np.array(y_val)


if __name__ == "__main__":
    # Preprocess training data
    train_folder = 'Final Project/Project_dataset/train'
    x_train, y_train = preprocess_images(train_folder)

    # Preprocess validation data
    val_folder = 'Final Project/Project_dataset/val'
    label_file = "Final Project/Project_dataset/val/labels.txt"
    x_val, y_val = load_validation_data(val_folder, label_file)
    #print(len(x_val), len(y_val))
    #print(x_val[0].shape, x_train[0].shape)

    # Save data using pickle
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((x_train, y_train, x_val, y_val), f)
    print("Data preprocessing complete and saved to 'processed_data.pkl'.")