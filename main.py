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
        mask = cv2.imread(msk_path) 

        # Read Image
        img_path = 'training_dataset/image/' + str(image_no) + '.png'
        image = cv2.imread(img_path)


        # # Image Processing
        out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out_image = cv2.GaussianBlur(out_image, (7, 7), 0)
        
        # otsu method
        __, mask1 = cv2.threshold(out_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask2 = cv2.bitwise_not(mask1)
        # watershed
        mask3 = watershed(image)
        mask4 = cv2.bitwise_not(mask3)

        # Apply masks to the image
        unmasked_pixels_mask1 = cv2.bitwise_and(image, image, mask=mask1)
        unmasked_pixels_mask2 = cv2.bitwise_and(image, image, mask=mask2)
        unmasked_pixels_mask3 = cv2.bitwise_and(image, image, mask=mask3)
        unmasked_pixels_mask4 = cv2.bitwise_and(image, image, mask=mask4)

        # compute Score Using RGB channel
        mask1_RGB_score = compute_RGB_score(image, mask1)
        mask2_RGB_score = compute_RGB_score(image, mask2)
        mask3_RGB_score = compute_RGB_score(image, mask3)
        mask4_RGB_score = compute_RGB_score(image, mask4)


        # compute score using smoothness
        mask1_smooth_score = compute_smoothness(unmasked_pixels_mask1)
        mask2_smooth_score = compute_smoothness(unmasked_pixels_mask2)
        mask3_smooth_score = compute_smoothness(unmasked_pixels_mask3)
        mask4_smooth_score = compute_smoothness(unmasked_pixels_mask4)

        mask1_score = mask1_RGB_score + mask1_smooth_score/10
        mask2_score = mask2_RGB_score + mask2_smooth_score/10
        mask3_score = mask3_RGB_score + mask3_smooth_score/10
        mask4_score = mask4_RGB_score + mask4_smooth_score/10

        scores = [mask1_score, mask2_score, mask3_score, mask4_score]
        max_score = max(scores)

        if max_score == mask1_score:
            Final_mask = mask1
            Score_list[0] += 1
            cv2.imwrite('output/output_' + get_image_name(img_path) + '.jpg', mask1)
        elif max_score == mask2_score:
            Score_list[1] += 1
            Final_mask = mask2
            cv2.imwrite('output/output_' + get_image_name(img_path) + '.jpg', mask2)
        elif max_score == mask3_score:
            Score_list[2] += 1
            Final_mask = mask3
            cv2.imwrite('output/output_' + get_image_name(img_path) + '.jpg', mask3)
        elif max_score == mask4_score:
            Score_list[3] += 1
            Final_mask = mask4
            cv2.imwrite('output/output_' + get_image_name(img_path) + '.jpg', mask4)
            
        
        iou = Evaluate(Final_mask, mask)
        Total_Score += iou
        cv2.imwrite('output/output_' + get_image_name(img_path) + '.jpg', Final_mask)
        print(f"{image_no}.png IoU Score: {iou:.2f}")


    print(f"Average IoU Score: {Total_Score / image_num:.2f}")
    print(f"mask1: {Score_list[0]} | mask2: {Score_list[1]} | mask3: {Score_list[2]} | mask4: {Score_list[3]}")

if __name__ == "__main__":
    main()