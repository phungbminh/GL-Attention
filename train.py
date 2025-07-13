import argparse
import torch
from trainer import DatasetTrainer
from backbone import ResNet18, VGG16
from attention import CBAMBlock, BAMBlock, scSEBlock
from attention.GLSABlock import GLSABlock
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
                        choices=["STL10", "Caltech101", "Caltech256", "Oxford-IIIT Pets", "HAM10000", "isic-2018-task-3"],
                        help="Choose dataset to train on (default: STL10)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for resizing (default: 224)")
    parser.add_argument("--backbone", type=str, default="VGG16",
                        choices=["VGG16", "ResNet18"],
                        help="Choose the backbone model (default: VGG16)")
    parser.add_argument("--attention", type=str, default="CBAM",
                        choices=["CBAM", "BAM", "scSE", "none"],
                        help="Choose an attention mechanism or none (default: CBAM)")
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

    # Initialize train and test datasets
    logger.info("Loading dataset...")
    dataset = GeneralDataset(args.dataset, args.dataset_path)
    train_dataset, test_dataset = dataset.get_splits(val_size=0.2, seed=args.random_seed, image_size=args.image_size)

    # Log dataset information
    logger.log_dataset_info(len(train_dataset), 0, len(test_dataset), dataset.num_classes)

    # Lấy danh sách nhãn từ train_ds
    train_labels = [train_dataset.lbls[i] for i in range(len(train_dataset))]

    class_counts = np.bincount(train_labels, minlength=train_dataset.num_classes)
    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[label] for label in train_labels]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # = len(train_ds)
        replacement=True
    )

    # DataLoader cho train với sampler
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
     # Select backbone model
    logger.info("Creating model...")
    model = None
    if args.backbone == "VGG16":
        model = VGG16(pretrained=args.pre_train, attn_type=args.attention, num_heads=8, num_classes=dataset.num_classes)
        # Replace with configurable GLSA block if attention is enabled and using non-original strategies
        if args.attention != 'none' and (args.fusion_strategy != 'original' or args.residual_strategy != 'original'):
            # Get VGG16 feature channels (assuming layer before classifier)
            channels = 512  # VGG16 last conv layer channels
            configurable_block = GLSABlock(
                channels=channels,
                attn_type=args.attention,
                fusion_strategy=args.fusion_strategy,
                residual_strategy=args.residual_strategy
            )
            model.mha_block = configurable_block
            logger.info(f"Using configurable GLSA block with fusion: {args.fusion_strategy}, residual: {args.residual_strategy}")
    elif args.backbone == "ResNet18":
        model = ResNet18(pretrained=args.pre_train, attn_type=args.attention, num_heads=8, num_classes=dataset.num_classes)
        # Replace with configurable GLSA block if attention is enabled and using non-original strategies
        if args.attention != 'none' and (args.fusion_strategy != 'original' or args.residual_strategy != 'original'):
            # Get ResNet18 layer3 channels
            from torchvision.models import resnet18
            backbone = resnet18()
            channels = backbone.layer3[-1].conv2.out_channels  # 256 for ResNet18
            
            configurable_block = GLSABlock(
                channels=channels,
                attn_type=args.attention,
                fusion_strategy=args.fusion_strategy,
                residual_strategy=args.residual_strategy
            )
            model.mha_block = configurable_block
            logger.info(f"Using configurable GLSA block with fusion: {args.fusion_strategy}, residual: {args.residual_strategy}")
    
    # Log model information
    logger.log_model_info(model)
    
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
    }
    if model is not None:
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = DatasetTrainer(model, train_loader, test_loader, test_loader, configs, wb=True, logger=logger, verbose=args.verbose)
        # Start training
        trainer.train()
        logger.info("Training completed successfully!")
    else:
        logger.error("Model creation failed!")
        raise ValueError("Model is None")


if __name__ == "__main__":
    main()
