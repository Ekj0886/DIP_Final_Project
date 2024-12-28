import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops #type: ignore
from skimage.morphology import remove_small_objects #type: ignore


def get_file_names(directory):
    return sorted(
        [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))],
        key=lambda x: int(os.path.splitext(x)[0])  # Convert the name (without extension) to an integer
    )

def get_image_name(path):
    last_slash = path.rfind('/')
    last_dot = path.rfind('.')
    image_name = path[last_slash + 1:last_dot]
    return image_name

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

def vertical_line_score(image):

    # Apply the Sobel filter for vertical edges
    sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = np.abs(sobel_vertical)  # Take absolute values for magnitude

    # Normalize the Sobel output
    normalized = cv2.normalize(sobel_vertical, None, 0, 255, cv2.NORM_MINMAX)

    # Compute the score (sum of vertical features)
    score = np.sum(normalized)

    return score

def horizontal_line_score(image):

    # Apply the Sobel filter for horizontal edges
    sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_horizontal = np.abs(sobel_horizontal)  # Take absolute values for magnitude

    # Normalize the Sobel output
    normalized = cv2.normalize(sobel_horizontal, None, 0, 255, cv2.NORM_MINMAX)

    # Compute the score (sum of horizontal features)
    score = np.sum(normalized)

    return score

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
    horizontal_score = horizontal_line_score(masked_region_gray)
    vertical_score   = vertical_line_score(masked_region_gray)

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

def compute_smoothness(image):
    # Convert to grayscale (if it's an RGB image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to find the gradients in x and y directions
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Compute smoothness score as the average gradient magnitude
    smoothness_score = np.mean(grad_magnitude)

    return smoothness_score

def compute_RGB_score(image, mask):
    # Step to get average RGB values for unmasked pixels in mask1
    # Get the non-zero indices of mask1
    unmasked_indices_mask = np.nonzero(mask)

    # Extract RGB values for mask1
    R_values_mask = image[:, :, 2][unmasked_indices_mask]  # Red channel
    G_values_mask = image[:, :, 1][unmasked_indices_mask]  # Green channel
    B_values_mask = image[:, :, 0][unmasked_indices_mask]  # Blue channel

    # Compute the average RGB values for mask1
    R_mean_mask = np.mean(R_values_mask)
    G_mean_mask = np.mean(G_values_mask)
    B_mean_mask = np.mean(B_values_mask)

    return (2*B_mean_mask + G_mean_mask) / R_mean_mask

def watershed(image):
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Threshold to create a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Remove noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 4: Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 5: Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Step 6: Unknown region (border)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 7: Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so the background is not 0
    markers = markers + 1

    # Mark the unknown region with zero
    markers[unknown == 255] = 0

    # Step 8: Apply the watershed algorithm
    markers = cv2.watershed(image, markers)

    # Step 9: Generate binary output (e.g., foreground regions only)
    binary_output = np.zeros_like(markers, dtype=np.uint8)
    binary_output[markers > 1] = 255  # Regions with markers > 1 are foreground

    return binary_output


def post_process_mask(binary_mask):
    # Remove small objects
    binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=500)
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return binary_mask


def white_balance_gray_world(image):
    # Split image into RGB channels
    B, G, R = cv2.split(image)
    
    # Compute the average for each channel
    B_avg, G_avg, R_avg = np.mean(B), np.mean(G), np.mean(R)
    
    # Scale each channel by the average
    K = (B_avg + G_avg + R_avg) / 3
    B = cv2.normalize(B * (K / B_avg), None, 0, 255, cv2.NORM_MINMAX)
    G = cv2.normalize(G * (K / G_avg), None, 0, 255, cv2.NORM_MINMAX)
    R = cv2.normalize(R * (K / R_avg), None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge the channels back
    balanced_image = cv2.merge([B, G, R]).astype(np.uint8)
    return balanced_image