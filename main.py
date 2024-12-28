import cv2
import os
import numpy as np
from utils import *


def main():
    Total_Score = 0
    Score_list = [0, 0, 0, 0]

    # Directories for images and masks
    # image_dir = 'water_v1/water_v1/JPEGImages/ADE20K'
    # mask_dir = 'water_v1/water_v1/Annotations/ADE20K'
    # output_dir = 'output'
    image_dir = 'testing_dataset/image'
    mask_dir = 'testing_dataset/mask'
    output_dir = 'testing_dataset/output'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted lists of image and mask files
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Ensure there are equal numbers of images and masks
    if len(image_files) != len(mask_files):
        print("Error: The number of images and masks does not match.")
        return

    for img_file, msk_file in zip(image_files, mask_files):
        # Construct paths for the image and mask
        img_path = os.path.join(image_dir, img_file)
        msk_path = os.path.join(mask_dir, msk_file)

        # Read mask and image
        origin_mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale
        image = cv2.imread(img_path)

        if origin_mask is None or image is None:
            print(f"Warning: Missing image or mask for file {img_file} or {msk_file}")
            continue

        # Preprocessing
        out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out_image = cv2.GaussianBlur(out_image, (7, 7), 0)

        # Otsu method
        _, mask1 = cv2.threshold(out_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask2 = cv2.bitwise_not(mask1)

        # Watershed
        mask3 = watershed(image)
        mask4 = cv2.bitwise_not(mask3)

        # Append to mask list
        mask_list = [mask1, mask2, mask3, mask4]

        max_score = -1
        Final_mask = None

        for i, mask in enumerate(mask_list):
            # Apply the mask to the image
            unmasked_pixels = cv2.bitwise_and(image, image, mask=mask)

            # Compute RGB score
            rgb_score = compute_RGB_score(image, mask)

            # Compute smoothness score
            smooth_score = compute_smoothness(unmasked_pixels)

            # Total score for this mask
            mask_score = rgb_score + smooth_score / 10

            # Update the maximum score and the final mask
            if mask_score > max_score:
                max_score = mask_score
                Final_mask = mask
                max_score_index = i

        # Update the Score_list for the selected mask
        Score_list[max_score_index] += 1

        # Save the final mask as an output image
        output_filename = f'output_{os.path.splitext(img_file)[0]}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, Final_mask)

        # Evaluate the IoU score
        iou = Evaluate(Final_mask, origin_mask)
        Total_Score += iou

        print(f"{img_file} IoU Score: {iou:.2f}")

    print(f"\nAverage IoU Score: {Total_Score / len(image_files):.2f}")
    print("================================================")
    print(f"| mask1: {Score_list[0]} | mask2: {Score_list[1]} | mask3: {Score_list[2]} | mask4: {Score_list[3]} |")
    print("================================================")


if __name__ == "__main__":
    main()
