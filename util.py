import cv2
import numpy as np

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



def Label(image):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    print(f"Number of connected components: {num_labels - 1}")

    # Iterate through each region
    for label in range(1, num_labels):  # Start from 1 to skip the background
        # Create a mask for the current component
        component_mask = (labels == label).astype(np.uint8) * 255
    
        # Extract pixel values of the current region
        region_pixels = image[labels == label]
        # print(f"Region {label}:")
        # print(f" - Pixel values: {region_pixels}")
        # print(f" - Area: {stats[label, cv2.CC_STAT_AREA]}")
        # print(f" - Centroid: {centroids[label]}")
    
        # Display the component (optional)
        # cv2.imshow(f'Region {label}', component_mask)
        # cv2.waitKey(0)


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