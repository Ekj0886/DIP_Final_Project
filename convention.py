import cv2
import os
import numpy as np
from sklearn.cluster import KMeans #type: ignore
from utils import *

train_image_dir = 'training_dataset/image/'
train_mask_dir = 'training_dataset/mask/'

test_image_dir = 'testing_dataset/image/'
test_mask_dir = 'testing_dataset/mask/'


output_dir = 'testing_dataset/output/'
seg_img_dir = 'seg_img/' 

def Segment():
    image_files = get_file_names(test_image_dir)
    mask_files = get_file_names(test_mask_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load precomputed feature distributions (for example, from a JSON file)
    # distributions = load_distributions('feature_distributions.json')

    iou_scores = []  # List to store IoU scores for each image
    
    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(test_image_dir, img_name)
        mask_path = os.path.join(test_mask_dir, mask_name)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        mask_watershed = watershed(image)
        cv2.imwrite(os.path.join('watershed', img_name), mask_watershed)

        _, regions = segment_image_kmeans_with_spatial(image, K=10)
        
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        

        for i, region_pixels in enumerate(regions):
            # Extract features for the region
            features = extract_region_features(region_pixels, image)
            all_features.append(features)
            

            

        # Calculate IoU score for the image
        iou = Evaluate(binary_mask, mask)
        iou_scores.append(iou)
        print(f"IoU for {img_name}: {iou:.2f}")
    
    # Calculate the average IoU score for all test images
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU for all test images: {mean_iou:.2f}")


if __name__ == "__main__":
    Segment()