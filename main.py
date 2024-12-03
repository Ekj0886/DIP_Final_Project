import cv2
import numpy as np
from util import get_image_name, Evaluate, compute_smoothness, compute_RGB_score, watershed
    
import cv2
import numpy as np


def main():

    Total_Score = 0
    image_num = 80

    Score_list = [0, 0, 0, 0]

    for image_no in range(1, image_num+1):
        # Read Mask
        msk_path = 'training_dataset/mask/'+ str(image_no) + '.png'
        origin_mask = cv2.imread(msk_path) 

        # Read Image
        img_path = 'training_dataset/image/' + str(image_no) + '.png'
        image = cv2.imread(img_path)


        # Preprocessing
        out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out_image = cv2.GaussianBlur(out_image, (7, 7), 0)

        # otsu method
        __, mask1 = cv2.threshold(out_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask2 = cv2.bitwise_not(mask1)
        # watershed
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
        output_path = f'output/output_{get_image_name(img_path)}.jpg'
        cv2.imwrite(output_path, Final_mask)

        # Evaluate the IoU score
        iou = Evaluate(Final_mask, origin_mask)
        Total_Score += iou

        print(f"{image_no}.png IoU Score: {iou:.2f}")


    print(f"\nAverage IoU Score: {Total_Score / image_num:.2f}")
    print("================================================")
    print(f"| mask1: {Score_list[0]} | mask2: {Score_list[1]} | mask3: {Score_list[2]} | mask4: {Score_list[3]} |")
    print("================================================")

    
if __name__ == "__main__":
    main()