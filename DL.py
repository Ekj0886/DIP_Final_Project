import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn.functional import interpolate
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

# Dataset Class
class WaterSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Load RGB image and single-channel mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        original_size = image.size  # Save original dimensions (width, height)

        # Apply transformations
        if self.transform_image and self.transform_mask:
            seed = np.random.randint(2147483647)  # Generate a random seed
            torch.manual_seed(seed)
            image = self.transform_image(image)
            torch.manual_seed(seed)
            mask = self.transform_mask(mask)

        return image, mask, original_size

# Simple U-Net Model
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec4 = self.dec4(torch.cat([interpolate(enc4, scale_factor=2, mode='bilinear', align_corners=False), enc3], dim=1))
        dec3 = self.dec3(torch.cat([interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False), enc2], dim=1))
        dec2 = self.dec2(torch.cat([interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False), enc1], dim=1))
        out = self.final(interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))

        return torch.sigmoid(out)

# IoU Calculation
def compute_iou(pred_mask, true_mask):
    pred_mask = (pred_mask > 0.5).int()
    true_mask = (true_mask > 0.5).int()

    intersection = (pred_mask & true_mask).float().sum((1, 2))
    union = (pred_mask | true_mask).float().sum((1, 2))

    iou = (intersection / (union + 1e-6)).mean()
    return iou

# Paths
train_image_dir = "training_dataset/image"
train_mask_dir = "training_dataset/mask"
test_image_dir = "testing_dataset/image"
test_mask_dir = "testing_dataset/mask"
# test_image_dir = "water_v1/water_v1/JPEGImages/ADE20K"
# test_mask_dir = "water_v1/water_v1/Annotations/ADE20K"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Train():
    # Transforms
    transform_image = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((480, 480)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((480, 480)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    # Cross-validation Setup
    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Training and Testing Setup
    criterion = nn.BCELoss()

    # Store IoU for each fold
    iou_scores = []
    best_model = None  # Variable to store the best model

    # Separate Training and Save Model
    num_epochs = 50
    for fold, (train_idx, val_idx) in enumerate(kf.split(os.listdir(train_image_dir))):
        print(f"Starting fold {fold + 1}/{k_folds}")

        model = SimpleUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Define scheduler: Reduce LR when validation loss plateaus for 3 epochs
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        train_subset = torch.utils.data.Subset(WaterSegmentationDataset(train_image_dir, train_mask_dir, transform_image, transform_mask), train_idx)
        val_subset = torch.utils.data.Subset(WaterSegmentationDataset(train_image_dir, train_mask_dir, transform_image, transform_mask), val_idx)

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for images, masks, _ in train_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                outputs = interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # Evaluate on the validation set after each epoch
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)

                    outputs = model(images)
                    outputs = interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            # Call the scheduler.step() to adjust the learning rate based on validation loss
            scheduler.step(val_loss)

            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Evaluate IoU on validation set for the current fold
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                outputs = interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

                val_iou += compute_iou(outputs, masks)

        avg_iou = val_iou / len(val_loader)
        print(f"Fold {fold + 1} - Validation IoU: {avg_iou:.4f}")
        iou_scores.append(avg_iou)

        # Save model for each fold
        model_path = f"model_fold_{fold + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

        # Update the best model based on the highest IoU
        if best_model is None or avg_iou > max(iou_scores[:-1]):  # Check if current fold has the best IoU
            best_model = model
            best_iou = avg_iou

    # After all folds are completed, save the best model as 'model.pth'
    print(f"Best fold: {iou_scores.index(best_iou) + 1} with IoU: {best_iou:.4f}")
    torch.save(best_model.state_dict(), "model.pth")
    print(f"Best model saved as 'model.pth'")

def find_mask_file(image_name, mask_dir):
    base_name = os.path.splitext(image_name)[0]
    for mask_file in os.listdir(mask_dir):
        if os.path.splitext(mask_file)[0] == base_name:
            return os.path.join(mask_dir, mask_file)
    raise FileNotFoundError(f"No corresponding mask found for image: {image_name}")


def Test():
    # Testing and IoU Calculation
    output_dir = "testing_dataset/output"
    os.makedirs(output_dir, exist_ok=True)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("model.pth"))  # Load the final retrained model
    model.eval()

    ious = []
    with torch.no_grad():
        test_images = os.listdir(test_image_dir)
        for idx, image_name in enumerate(test_images):
            image_path = os.path.join(test_image_dir, image_name)
            mask_path = find_mask_file(image_name, test_mask_dir)

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])

            transform_test_mask = transforms.Compose([
                transforms.ToTensor(),
            ])

            image = transform(image).to(device)
            mask = transform_test_mask(mask).to(device)

            outputs = model(image.unsqueeze(0))
            original_height, original_width = mask.shape[1:]

            preds = interpolate(outputs, size=(original_height, original_width), mode="bilinear", align_corners=False)
            preds = (preds > 0.5).float()

            output_path = os.path.join(output_dir, image_name)
            pred_image = transforms.ToPILImage()(preds.squeeze(0).cpu())
            pred_image.save(output_path)

            iou = compute_iou(preds.squeeze(1).cpu(), mask.unsqueeze(0).cpu())
            ious.append(iou.item())

            # Print IoU for each image
            print(f"Image {image_name} IoU: {iou.item():.4f}")

    print(f"Average IoU: {np.mean(ious):.4f}")

if __name__ == "__main__":
    print('Mode <all> or <train> or <test>: ')
    mode = input()
    if mode == 'all':
        Train()
        Test()
    elif mode == 'train':
        Train()
    elif mode == 'test':
        Test()
    else:
        print('Invalid mode')
