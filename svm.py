import cv2
import os
import numpy as np
from sklearn.svm import SVC #type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.model_selection import StratifiedKFold #type: ignore
from utils import *
import joblib #type: ignore

train_image_dir = 'training_dataset/image/'
train_mask_dir = 'training_dataset/mask/'
test_image_dir = 'testing_dataset/image/'
test_mask_dir = 'testing_dataset/mask/'
output_dir = 'testing_dataset/output/'
seg_img_dir = 'seg_img/' 

svm_model = None  # Initialize the model as global variable

def Train():
    print('== Train SVM mode')
    print('== Extract Features')
    global svm_model  # Indicate that we want to modify the global svm_model variable
    image_files = get_file_names(train_image_dir)
    mask_files = get_file_names(train_mask_dir)
    
    all_features = []  # To store feature vectors
    all_labels = []    # To store corresponding labels (0 = non-water, 1 = water)

    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(train_image_dir, img_name)
        mask_path = os.path.join(train_mask_dir, mask_name)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print('    img ' + str(img_name) + ' load ' + str(image.shape))
        
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        out_image, regions = segment_image_kmeans_with_spatial(image, K=20)
        
        water_regions, non_water_regions = split_regions_by_mask(regions, mask)
    
        for region_pixels in water_regions:
            features = extract_region_features(region_pixels, image)
            all_features.append(features)
            all_labels.append(1)

        for region_pixels in non_water_regions:
            features = extract_region_features(region_pixels, image)
            all_features.append(features)
            all_labels.append(0)
        
        output_img_path = os.path.join(seg_img_dir, f"{os.path.splitext(img_name)[0]}_segmented.png")
        cv2.imwrite(output_img_path, out_image)


    X = np.array(all_features)
    y = np.array(all_labels)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler for later use in testing
    joblib.dump(scaler, 'scaler.pkl')
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print('== SVM Training')
    svm_model = SVC(kernel='poly', class_weight='balanced')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter tuning for SVM with polynomial kernel
    param_grid = {
        'C': [0.1, 1, 10],
        'degree': [2, 3, 4],
        'coef0': [0, 1, 10],
        'kernel': ['poly']
    }
    svm = SVC(class_weight='balanced')
    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model and validation accuracy
    svm_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    accuracy = svm_model.score(X_val, y_val)
    print(f"== SVM Validation Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model to a file
    joblib.dump(svm_model, 'svm_model.pkl')
    
    

def Test():
    print('== Test mode')
    # Load the trained model and scaler
    svm_model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    image_files = get_file_names(test_image_dir)
    mask_files = get_file_names(test_mask_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    iou_scores = []  # List to store IoU scores for each image
    
    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(test_image_dir, img_name)
        mask_path = os.path.join(test_mask_dir, mask_name)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (5, 5), 0)
    
        _, regions = segment_image_kmeans_with_spatial(image, K=20)
        
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for i, region_pixels in enumerate(regions):
            features = extract_region_features(region_pixels, image)
            scaled_features = scaler.transform([features])  # Scale features
            predicted_label = svm_model.predict(scaled_features)[0]

            for x, y in region_pixels:
                binary_mask[x, y] = predicted_label * 255
        
        # Save the binary mask output
        output_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(output_img_path, binary_mask)
        
        # Calculate IoU score for the image
        iou = Evaluate(binary_mask, mask)
        iou_scores.append(iou)
        print(f"IoU for {img_name}: {iou:.2f}")
    
    # Calculate the average IoU score for all test images
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU for all test images: {mean_iou:.2f}")


if __name__ == "__main__":
    print('Mode <train> or <test>: ')
    mode = input()

    if mode == 'train':
        Train()
    elif mode == 'test':
        Test()
    else:
        print('Invalid mode')
