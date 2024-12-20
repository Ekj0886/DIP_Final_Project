import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops


def get_file_names(directory):
    return sorted(
        [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))],
        key=lambda x: int(os.path.splitext(x)[0])  # Convert the name (without extension) to an integer
    )

def segment_image_kmeans_with_spatial(image, K=7):

    # Step 1: Convert the image to RGB (if it's in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Get the image shape and reshape it to a 2D array of pixels (RGB values)
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape((-1, 3)).astype(np.float32)

    # Step 3: Generate spatial features (x, y coordinates for each pixel)
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    spatial_features = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)

    # Step 4: Concatenate the RGB and spatial features (x, y) into a single feature vector
    pixels_with_spatial = np.concatenate((pixels, spatial_features), axis=1)

    # Ensure data type is np.float32
    pixels_with_spatial = pixels_with_spatial.astype(np.float32)

    # Step 5: Apply K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels_with_spatial, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Step 6: Map the cluster labels back to the image
    clustered_image = labels.reshape((h, w))

    # Step 7: Generate a segmented image using the cluster centers
    segmented_image = np.zeros_like(image_rgb)
    for i in range(K):
        segmented_image[clustered_image == i] = np.uint8(centers[i, :3])  # Use only RGB values for visualization

    # Convert the segmented image back to BGR for visualization
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    # Collect pixel indices for each region
    region_pixels = []
    for i in range(K):
        pixels_in_region = np.argwhere(clustered_image == i)
        region_pixels.append([tuple(p) for p in pixels_in_region])
        # print(f"Region {i + 1}: {len(pixels_in_region)} pixels")

    return segmented_image, np.array(region_pixels, dtype=object)


def split_regions_by_mask(regions, mask):
    water_regions = []
    non_water_regions = []

    for region in regions:
        water_pixels = []
        non_water_pixels = []
        
        for pixel in region:
            x, y = pixel
            if mask[x, y] == 255:
                water_pixels.append((x, y))
            else:
                non_water_pixels.append((x, y))
        
        if water_pixels:
            water_regions.append(np.array(water_pixels))
        if non_water_pixels:
            non_water_regions.append(np.array(non_water_pixels))

    # return np.array(water_regions, dtype=object), np.array(non_water_regions, dtype=object)
    return water_regions, non_water_regions

def extract_region_features(region_pixels, image):
    # Extract RGB values for the region
    region_rgb = np.array([image[x, y] for x, y in region_pixels])
    
    # Compute mean RGB values
    r_mean = np.mean(region_rgb[:, 0])
    g_mean = np.mean(region_rgb[:, 1])
    b_mean = np.mean(region_rgb[:, 2])

    # Convert full image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create the grayscale region using the region's pixel coordinates
    region_gray = np.array([grayscale_image[x, y] for x, y in region_pixels])

    # Create a binary mask for the region and apply it to the grayscale image
    region_mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
    for x, y in region_pixels:
        region_mask[x, y] = 1  # Note that (x, y) might need to be swapped to (y, x) for correct indexing

    # Apply the mask to the grayscale image (extract region from the grayscale image)
    masked_region_gray = grayscale_image * region_mask

    # Compute GLCM on the masked grayscale region
    distances = [1]  # Pixel distance
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles (0°, 45°, 90°, 135°)
    glcm = graycomatrix(masked_region_gray, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract GLCM properties
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    # Combine features into a feature vector
    feature_vector = np.array([r_mean, g_mean, b_mean, contrast, homogeneity, energy, correlation])
    
    return feature_vector


def Evaluate(image: np.ndarray, mask: np.ndarray) -> float:
    # Check if the mask has more than one channel (e.g., BGR) and convert it to grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # If the mask is already binary, you may skip thresholding
    if np.max(mask) > 1:  # Check if the mask contains more than two values
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Binarizing the image and mask
    image_binary = image // 255  
    mask_binary = mask // 255

    intersection = np.logical_and(image_binary, mask_binary).sum()
    union = np.logical_or(image_binary, mask_binary).sum()

    if union == 0:
        return 0.0  # Avoid division by zero

    iou = intersection / union
    return iou