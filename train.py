import argparse
import torch
from trainer import DatasetTrainer
from backbone.models import create_model, SUPPORTED_BACKBONES
from datasets import GeneralDataset
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import random, numpy as np, torch
from torch.utils.data import WeightedRandomSampler
from utils.logger import setup_logger, get_logger

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across modules.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_arguments():
    """  
    Parse arguments for training configuration.  
    """
    parser = argparse.ArgumentParser(
        description="Train model with various backbones, attention mechanisms, and configurations.")

    # Dataset and model arguments  
    parser.add_argument("--dataset", type=str, default="HAM10000",
                        choices=["HAM10000", "isic-2018-task-3"],
                        help="Choose medical imaging dataset (default: HAM10000)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for resizing (default: 224)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=SUPPORTED_BACKBONES,
                        help="Choose the backbone model (default: resnet18)")
    parser.add_argument("--attention", type=str, default="CBAM",
                        choices=["CBAM", "none"],
                        help="Choose attention mechanism: CBAM or none (default: CBAM)")
    parser.add_argument("--fusion_strategy", type=str, default="original",
                        choices=["original", "learnable_gate", "weighted_add", "channel_attn", "spatial_attn", "cross_attn"],
                        help="Choose fusion strategy for GLSA block (default: original)")
    parser.add_argument("--residual_strategy", type=str, default="original",
                        choices=["original", "learnable_gate", "progressive", "drop_path"],
                        help="Choose residual connection strategy (default: original)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader (default: 0)")

    # Training hyperparameters  
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate (default: 1e-7)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    parser.add_argument("--optimizer", type=str, default="RAdam",
                        choices=["RAdam", "Adam", "SGD", "AdamW"], help="Optimizer to use (default: RAdam)")
    parser.add_argument("--lr_scheduler", type=str, default="ReduceLROnPlateau",
                        choices=["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "CosineWarmup","None"],
                        help="Learning rate scheduler to use (default: ReduceLROnPlateau)")
    parser.add_argument("--max_epoch", type=int, default=10, help="Maximum number of training epochs (default: 10)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Maximum number of training epochs (default: 10)")

    # Device and reproducibility  
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device for training (default: cuda)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Logging and checkpoints  
    parser.add_argument("--wandb_project", type=str, default="test-project",
                        help="WandB project name (default: test-project)")
    parser.add_argument("--wandb_run", type=str, default="run-v1",
                        help="WandB run name (default: run-v1)")
    parser.add_argument("--wandb_key", type=str, default="run-v1",
                        help="WandB run api key")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth",
                        help="Path to save the best model (default: best_model.pth)")
    parser.add_argument("--pre_train", action="store_true",help="Enable pre-training mode")
    parser.add_argument("--dataset_path", type=str, default="best_model.pth",
                        help="Path to save the best model (default: best_model.pth)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging (show batch-level metrics)")
    
    # Loss function arguments
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["focal", "cross_entropy", "class_balanced", "label_smoothing"],
                        help="Loss function type (default: focal)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha parameter (default: 0.25)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (default: 2.0)")

    return parser.parse_args()


def main():
    # Parse arguments  
    args = parse_arguments()

    # Setup logger 
    import logging
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(log_dir="logs", project_name="GL-Attention", 
                         console_level=console_level, file_level=logging.DEBUG)
    
    # Set seed
    set_seed(args.random_seed)
    
    # Log configuration
    config_dict = vars(args)
    logger.log_config(config_dict, "Training Configuration")
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Print system info for debugging
    print(f"System Info:")
    print(f"   Device: {device}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num workers: {args.num_workers}")
    print(f"   Image size: {args.image_size}x{args.image_size}")
    print(f"   Max epochs: {args.max_epoch}")
    print()

    # Initialize train and test datasets
    logger.info("Loading dataset...")
    print(f"Loading {args.dataset} dataset...")
    dataset = GeneralDataset(args.dataset, args.dataset_path)
    
    print("Creating train/validation splits...")
    train_dataset, test_dataset = dataset.get_splits(val_size=0.2, seed=args.random_seed, image_size=args.image_size)

    # Log dataset information
    logger.log_dataset_info(len(train_dataset), 0, len(test_dataset), dataset.num_classes)

    # Create weighted sampler for handling class imbalance
    print("Creating WeightedRandomSampler for class balance...")
    train_sampler = train_dataset.get_sampler(method='sqrt')
    logger.info("Using WeightedRandomSampler with sqrt weighting for class balance")

    # DataLoader with weighted sampler
    print("Setting up data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers
    )

    #Create DataLoaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers
    # )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    # Create model with unified backbone system
    print(f"Creating {args.backbone} model with {args.attention} attention...")
    logger.info(f"Creating {args.backbone} model with {args.attention} attention...")
    
    model = create_model(
        backbone_name=args.backbone,
        pretrained=args.pre_train,
        num_classes=dataset.num_classes,
        attn_type=args.attention,
        fusion_strategy=args.fusion_strategy,
        residual_strategy=args.residual_strategy,
        num_heads=8,
        reduction_ratio=16
    )
    print("Model created successfully!")
    
    # Log model information
    logger.log_model_info(model)
    logger.info(f"Feature channels at GLSA integration: {model.get_feature_channels()}")
    
    # Log fusion strategy if using non-original
    if args.fusion_strategy != 'original' or args.residual_strategy != 'original':
        logger.log_fusion_strategy(args.fusion_strategy, args.residual_strategy)
    # Create run name with fusion strategy info
    run_name = f"{args.wandb_run}_{args.fusion_strategy}_{args.residual_strategy}"
    if args.attention != 'none':
        run_name += f"_{args.attention}"
    
    configs = {
        "device": device,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "lr_scheduler": args.lr_scheduler,
        "max_epoch_num": args.max_epoch,
        "checkpoint_path": args.checkpoint_path,
        "wandb_api_key": args.wandb_key,
        "project_name": args.wandb_project,
        "run_name": run_name,
        "early_stopping_patience": args.early_stopping_patience,
        # Add fusion strategy info for tracking
        "fusion_strategy": args.fusion_strategy,
        "residual_strategy": args.residual_strategy,
        # Loss function configuration
        "loss_type": args.loss_type,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
    }
    if model is not None:
        # Initialize trainer
        print("Initializing trainer...")
        logger.info("Initializing trainer...")
        trainer = DatasetTrainer(model, train_loader, test_loader, test_loader, configs, wb=True, logger=logger, verbose=args.verbose)
        
        # Start training
        print("Starting training process...")
        trainer.train()
        print("Training completed successfully!")
        logger.info("Training completed successfully!")
    else:
        logger.error("Model creation failed!")
        raise ValueError("Model is None")


if __name__ == "__main__":
    main()
