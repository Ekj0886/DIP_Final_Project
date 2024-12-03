import cv2
import numpy as np

def get_image_name(path):
    last_slash = path.rfind('/')
    last_dot = path.rfind('.')
    image_name = path[last_slash + 1:last_dot]
    return image_name


# Usage: read in image and mask, use Evaluate(image, mask) to get iou score
# Note that image should be binary 
# e.g.  print(f"IoU Score: {Evaluate(out_image, mask):.2f}")
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
