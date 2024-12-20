import cv2
import numpy as np
import os

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


# Usage: read in image and mask, use Evaluate(image, mask) to get iou score
# Note that image should be binary 
# e.g.  print(f"IoU Score: {Evaluate(image, mask):.2f}")
def Evaluate(image: np.ndarray, mask: np.ndarray) -> float:
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    __, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image_binary = image // 255  
    mask_binary = mask // 255

    intersection = np.logical_and(image_binary, mask_binary).sum()
    union = np.logical_or(image_binary, mask_binary).sum()

    if union == 0:
        return 0.0  # Avoid division by zero

    iou = intersection / union
    return iou


def compute_smoothness(image):
    """
    Compute the smoothness score of the image by calculating the average gradient magnitude.
    
    Args:
    - image (numpy array): 3D array (H, W, 3) in RGB or BGR format.
    
    Returns:
    - smoothness_score (float): Average gradient magnitude.
    """
    # Convert to grayscale if it's a color image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to find gradients in x and y directions
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize gradient magnitude to scale between 0 and 1
    grad_magnitude_norm = grad_magnitude / np.max(grad_magnitude)

    # Compute smoothness score as the average gradient magnitude
    smoothness_score = np.mean(grad_magnitude_norm)

    return smoothness_score


def compute_RGB_score(image, mask):
    """
    Compute the normalized RGB cost function incorporating means and variances.
    
    Args:
    - image (numpy array): 3D array (H, W, 3) in RGB or BGR format.
    - mask (numpy array): 2D array (H, W) indicating the region of interest.
    
    Returns:
    - cost (float): Normalized RGB cost value.
    """
    unmasked_indices = np.nonzero(mask)

    # Extract RGB values for the unmasked region
    R_values = image[:, :, 2][unmasked_indices]  # Red channel
    G_values = image[:, :, 1][unmasked_indices]  # Green channel
    B_values = image[:, :, 0][unmasked_indices]  # Blue channel

    # Compute mean values
    R_mean = np.mean(R_values)
    G_mean = np.mean(G_values)
    B_mean = np.mean(B_values)

    # Compute variances
    R_variance = np.var(R_values)
    G_variance = np.var(G_values)
    B_variance = np.var(B_values)

    # Avoid division by zero
    R_mean = R_mean if R_mean != 0 else 1

    # Normalize means and variances
    mean_max = max(R_mean, G_mean, B_mean, 1)
    var_max = max(R_variance, G_variance, B_variance, 1)

    R_mean_norm = R_mean / mean_max
    G_mean_norm = G_mean / mean_max
    B_mean_norm = B_mean / mean_max

    R_variance_inv = 1 / (R_variance + 1e-6)  # Inverse of variance to reward lower variance
    G_variance_inv = 1 / (G_variance + 1e-6)
    B_variance_inv = 1 / (B_variance + 1e-6)

    # Define a normalized RGB cost function
    weight_mean = 0.5
    weight_variance = 0.5

    cost = (
        weight_mean * (2 * B_mean_norm + G_mean_norm) / R_mean_norm
        + weight_variance * (R_variance_inv + G_variance_inv + B_variance_inv)
    )

    return cost


def compute_normalized_balance(image, mask, unmask_pixel, weight_smoothness=0.3, weight_RGB=0.7):
    smoothness_score = compute_smoothness(unmask_pixel)
    rgb_cost = compute_RGB_score(image, mask)

    # Normalize the combined components
    combined_cost = weight_smoothness * smoothness_score + weight_RGB * rgb_cost

    return combined_cost


def hsv_segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, binary = cv2.threshold(hsv[:, :, 1], 50, 255, cv2.THRESH_BINARY_INV)
    return binary


def opening(binary_images, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_images = []
    
    for binary_image in binary_images:
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        opened_images.append(opened_image)
    
    return opened_images


def closing(binary_images, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_images = []
    
    for binary_image in binary_images:
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        closed_images.append(closed_image)
    
    return closed_images

def DrawLine(mask):
    height, width = mask.shape

    # Define the grid spacing (e.g., 50 pixels apart for both horizontal and vertical)
    grid_spacing = 100

    # Draw vertical grid lines
    for x in range(0, width, grid_spacing):
        cv2.line(mask, (x, 0), (x, height), (0, 0, 0), thickness=1)
    
    # Draw horizontal grid lines
    for y in range(0, height, grid_spacing):
        cv2.line(mask, (0, y), (width, y), (0, 0, 0), thickness=1)

    return mask


def Label(image):
    # Threshold the image to ensure it's binary (0 or 255 values only)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components and their stats (including area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    
    # print(f"Number of connected components: {num_labels - 1}")  # Exclude background

    # Initialize a list to store the pixel areas of each component
    component_areas = []
    
    # Iterate through each connected component (skip the background)
    for label in range(1, num_labels):
        # Create a mask for the current component
        component_mask = (labels == label).astype(np.uint8) * 255

        # Extract pixel values for the current region
        region_pixels = image[labels == label]
        
        # Calculate the area (number of pixels) of the current region
        area = region_pixels.size
        
        # Append the area to the component_areas list
        component_areas.append(area)
    
    # Print the number of labels (regions) found, excluding background
    # print(f'Number of labels (regions): {num_labels - 1}')
    
    # Return the list of areas for each connected component
    return component_areas


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


import cv2
import numpy as np

def extract_features(image):
    """
    Extract pixel-level features (RGB and HSV) from an image.
    
    Args:
    - image: Input RGB image.

    Returns:
    - feature_matrix: A 2D array where each row contains the extracted features for a pixel.
    """
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract RGB and HSV values
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    H, S, V = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    height, width, _ = image.shape
    feature_matrix = np.zeros((height * width, 6))

    # Stack RGB and HSV features
    feature_matrix[:, 0] = R.flatten() / 255.0  # Normalize to [0, 1]
    feature_matrix[:, 1] = G.flatten() / 255.0
    feature_matrix[:, 2] = B.flatten() / 255.0
    feature_matrix[:, 3] = H.flatten() / 179.0  # Normalize Hue to [0, 1]
    feature_matrix[:, 4] = S.flatten() / 255.0  # Saturation
    feature_matrix[:, 5] = V.flatten() / 255.0  # Value

    return feature_matrix

def get_labels(mask):
    """
    Flatten the mask and return binary labels (0 or 1).
    """
    return mask.flatten() / 255.0  # Normalize mask to binary {0, 1}
