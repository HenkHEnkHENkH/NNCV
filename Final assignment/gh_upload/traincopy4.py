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

from modelcopy5 import Model

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
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            torch.clamp(targets, 0, self.num_classes - 1),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        mask = (targets != self.ignore_index).unsqueeze(1)
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask

        intersection = (probs * targets_one_hot).sum(dim=(0,2,3))
        cardinality = (probs + targets_one_hot).sum(dim=(0,2,3))

        dice = (2. * intersection + 1) / (cardinality + 1)
        dice_loss = 1 - dice.mean()
        
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
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")  # Increased
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
        A.RandomResizedCrop(size=(256, 512), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(p=0.2),
        A.CoarseDropout(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
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

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # Define the loss function
    # criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)  # Ignore the void class
    criterion = DiceCELoss(num_classes=19, ignore_index=255).to(device)
  
    encoder_prefixes = ("stem", "layer1", "layer2", "layer3", "layer4")

    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if name.startswith(encoder_prefixes):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    # 2. Initialize the optimizer with the sorted groups
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5}, # Group 0: ResNet Encoder
        {'params': decoder_params, 'lr': 1e-4}  # Group 1: Decoder, ASPP, Attention
    ], weight_decay=0.05)

    # 3. Setup OneCycleLR (Ensure max_lr order matches the groups above)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-5, 3e-4], # [Group 0 Max, Group 1 Max]
        epochs=args.epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.2,       
        div_factor=10,       
        final_div_factor=100 
    )


    # Assuming your physical batch size is 16 and you have ~2975 images
    # 
    # steps_per_epoch = len(train_dataset) // args.batch_size

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
            scheduler.step()  # Update LR after each batch

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
            current_step = (epoch + 1) * len(train_dataloader)
            wandb.log({
                "valid_loss": valid_loss,
                #"charts/learning_rate": current_lr,
                "epoch": epoch + 1
            }, step=current_step)
  # Zero out gradients for the next iteration

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
