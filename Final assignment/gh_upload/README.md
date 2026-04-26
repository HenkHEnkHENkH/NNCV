5LSM0 Cityscapes Semantic Segmentation
This repository contains a robust Attention-U-Net implementation for the Cityscapes dataset, specifically optimized for Domain Generalization from normal urban environments to hard adverse conditions.

Key Features
Architecture: ResNeXt-50 backbone with residual attention U-Net decoder.
Modules: Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context and Attention Gates for spatial pruning.
Loss: DiceCE Loss to handle class imbalance and ensure gradient stability.
Augmentations: Heavy weather simulation (fog, rain, snow) via Albumentations.

Requirements
PyTorch
Albumentations
WandB (Weights & Biases)

How to Run
To run the training on a Slurm-managed cluster using the provided container:
Configure Environment: Ensure your .env file contains your WANDB_API_KEY.
Submit the Job:
Bash
sbatch jobscript_slurm.sh
The jobscript_slurm.sh executes the main.sh entry point within an Apptainer/Singularity container.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh
