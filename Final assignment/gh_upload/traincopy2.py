"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser
from sched import scheduler
import numpy as np
import wandb
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
import torch.nn.functional as F
from PIL import Image

from modelcopy4 import Model

class DiceCELoss(nn.Module):
    """
    A hybrid loss that combines Cross Entropy (pixel accuracy) 
    with Dice Loss (region overlap).
    """
    def __init__(self, num_classes=19, ignore_index=255):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 1. Standard Cross Entropy
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        
        # 2. Dice Loss calculation
        # Filter out the ignore_index (255) so it doesn't ruin the Dice score
        mask = (targets != self.ignore_index)
        logits = logits.permute(0, 2, 3, 1)[mask] # Shape: (Pixels, Classes)
        targets = targets[mask]                   # Shape: (Pixels)

        if targets.numel() == 0:
            return ce_loss # Return only CE if the whole batch is 'ignore' pixels

        # Convert to probabilities and one-hot
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Intersection over Union logic
        intersection = torch.sum(probs * targets_one_hot, dim=0)
        cardinality = torch.sum(probs + targets_one_hot, dim=0)
        dice_loss = 1 - (2. * intersection + 1) / (cardinality + 1)
        
        # Weighting the two losses (hyperparameters)
        lambda_ce = 0.25
        lambda_dice = 1 - lambda_ce
        # Return the average of both
        return lambda_ce*ce_loss + lambda_dice*dice_loss.mean()
    
# Custom dataset class using albumentations for synchronized augmentations
class AugmentedCityscapes(Cityscapes):
    def __init__(self, root, split='train', mode='fine', target_type='semantic', transforms=None):
        # Only pass arguments Cityscapes actually expects
        super().__init__(root, split=split, mode=mode, target_type=target_type)
        self.apply_albumentations = transforms
    
    def __getitem__(self, idx):
        # 1. Get raw PIL images from base class
        image, target = super().__getitem__(idx)
        
        # 2. Convert to NumPy (Albumentations requirement)
        image = np.array(image)
        target = np.array(target)
        
        # 3. Apply Albumentations
        if self.apply_albumentations:
            augmented = self.apply_albumentations(image=image, mask=target)
            image = augmented['image']
            target = augmented['mask']
        
        # 4. CRITICAL: Ensure target is a Torch Long Tensor
        # Albumentations' ToTensorV2() handles the image, but the mask 
        # often needs an explicit cast to Long for segmentation losses.
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).long()
        else:
            target = target.long() 
            
        return image, target

# Mapping class IDs to train IDs

def convert_to_train_id(labels):
    # This uses the labels as indices to pick the correct train_id
    return mapping_tensor[labels.long()]
# Mapping train IDs to color

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
mapping_tensor = torch.full((256,), 255, dtype=torch.long)
for cls in Cityscapes.classes:
    mapping_tensor[cls.id] = cls.train_id
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")  # Reduced
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")  # Increased
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define synchronized augmentations using albumentations for training
    # TRAINING: Focus on variety and scale

    # TRAINING: RandomResizedCrop wants 'size'
    train_transforms = A.Compose([
        #A.RandomResizedCrop(size=(256, 512), scale=(0.5, 1.0), p=0.5),
        A.Resize(height=256, width=512, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

    # VALIDATION: Resize wants 'height' and 'width'
    valid_transforms = A.Compose([
        A.Resize(height=256, width=512), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

    # Load the dataset and make a split for training and validation
    train_dataset = AugmentedCityscapes(
        args.data_dir,
        split="train",
        transforms=train_transforms
    )

    valid_dataset = AugmentedCityscapes(
        args.data_dir,
        split="val",
        transforms=valid_transforms
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    class_weights = torch.tensor([
    1,   # Road 
    1,   # Sidewalk
    1,   # Building
    1,   # Wall
    1,   # Fence
    1,   # Pole
    1,  # Traffic Light 
    1,  # Traffic Sign 
    1,   # Vegetation
    1,   # Terrain
    1,   # Sky
    1,  # Person
    1,  # Rider
    1,   # Car
    1,  # Truck
    1,  # Bus
    1,  # Train
    1,  # Motorcycle
    1   # Bicycle
], dtype=torch.float32).to(device)
    
    # Define the loss function
    # criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)  # Ignore the void class
    criterion = DiceCELoss(num_classes=19, ignore_index=255).to(device)
    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Assuming your physical batch size is 16 and you have ~2975 images
    steps_per_epoch = len(train_dataset) // args.batch_size

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=1e-5,
        max_lr=args.lr,        # Stays at 0.001 every cycle
        step_size_up=20 * steps_per_epoch,
        mode='triangular2',     
        cycle_momentum=False
    )
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
    # 1. Map IDs (Do this on CPU or GPU, but Tensor mapping is best)
            labels = convert_to_train_id(labels) 
            
            # 2. Move to device
            images, labels = images.to(device), labels.to(device)
            
            # 3. Standardize shape for CrossEntropy (B, H, W)
            if labels.dim() == 4: # If it's (B, C, H, W)
                labels = labels.squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(dim=1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            # ... end of Validation loop ...
            valid_loss = sum(losses) / len(losses)

            # Optional: Log the new LR to see it on your WandB graphs
            current_lr = scheduler.get_last_lr()[0]
            current_step = (epoch + 1) * len(train_dataloader)
            wandb.log({
                "valid_loss": valid_loss,
                "charts/learning_rate": current_lr,
                "epoch": epoch + 1
            }, step=current_step)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
