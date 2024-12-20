import cv2
import numpy as np
import os
from util import *
import cv2
import numpy as np

train_image_dir = 'training_dataset/image/'
test_image_dir = 'testing_dataset/image/'
train_mask_dir = 'training_dataset/mask/'
test_mask_dir = 'testing_dataset/mask/'
train_output_dir = 'output/'
test_output_dir = 'testing_dataset/output/'

def main():
    dumb()
    # svm()

def svm():
    print('svm method')
    

def dumb():

    print("Input 'train' for train mode, 'test' for test mode: ")
    mode = input()
    if mode == 'train':
        image_dir = train_image_dir
        mask_dir  = train_mask_dir
        output_dir = train_output_dir
    else:
        image_dir = test_image_dir
        mask_dir = test_mask_dir
        output_dir = test_output_dir

    Total_Score = 0
    # image_num = 1
    image_num = sum(1 for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file)))
    image_file = get_file_names(image_dir)
    mask_file  = get_file_names(mask_dir)

    Score_list = np.zeros(6)

    # print(image_file)
    # print(mask_file)    

    for image_no in range(1, image_num+1):
        # Read Mask
        msk_path = mask_dir + mask_file[image_no-1] 
        origin_mask = cv2.imread(msk_path) 

        # Read Image
        img_path = image_dir + image_file[image_no-1]
        image = cv2.imread(img_path)

        # Preprocessing
        out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out_image = cv2.GaussianBlur(out_image, (7, 7), 0)
        

        # otsu method
        __, mask1 = cv2.threshold(out_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask2 = cv2.bitwise_not(mask1)
        # color space
        mask3 = hsv_segmentation(image)
        mask4 = cv2.bitwise_not(mask3)

        # Append to mask list
        mask_list = [mask1, mask2, mask3, mask4]
        
        mask_list = opening(mask_list)

        max_score = -1
        Final_mask = None

        for i, mask in enumerate(mask_list):
            # Apply the mask to the image
            unmasked_pixels = cv2.bitwise_and(image, image, mask=mask)

            # Compute RGB score
            # rgb_score = compute_RGB_score(image, mask)

            # # Compute smoothness score
            # smooth_score = compute_smoothness(unmasked_pixels) / 10

            # Total score for this mask
            mask_score = compute_normalized_balance(image, mask, unmasked_pixels)
            
            # print(f'rgb: {rgb_score:.2f} | smooth: {smooth_score:.2f}')
            # Update the maximum score and the final mask
            if mask_score > max_score:
                max_score = mask_score
                Final_mask = mask
                max_score_index = i


        # Update the Score_list for the selected mask
        Score_list[max_score_index] += 1

        # Save the final mask as an output image
        output_path = output_dir + f'{get_image_name(img_path)}.jpg'
        cv2.imwrite(output_path, Final_mask)

        # Evaluate the IoU score
        iou = Evaluate(Final_mask, origin_mask)
        Total_Score += iou

        print(f"{image_no}.png IoU Score: {iou:.2f} | max_score: {max_score:.2f} | mask: {max_score_index+1}")


        # Experiment on grid separation
        # Get the dimensions of the image
        
        # mask1 = DrawLine(mask1)
        # cv2.imwrite('output_image_with_grid.png', mask1)
        # object_list = Label(mask1)
        # print(len(object_list))


    print(f"\nAverage IoU Score: {Total_Score / image_num:.2f}")
    print("========================================================")
    print(f"| mask1: {Score_list[0]} | mask2: {Score_list[1]} | mask3: {Score_list[2]} | mask4: {Score_list[3]} |")
    print("========================================================")

    
if __name__ == "__main__":
    main()