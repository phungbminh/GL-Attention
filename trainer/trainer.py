import os
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR
)
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb  # Import wandb for logging
import csv
import math
from utils.logger import get_logger
from utils.losses import FocalLoss, get_loss_function

class EarlyStopping:
    """
    Early stops training when validation accuracy doesn't improve after a given patience.
    """
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_acc = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc <= self.best_acc + self.delta:
            self.counter += 1
            print(f"EarlyStoppingAccuracy counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0
class Trainer:
    """Base trainer class."""
    def __init__(self):
        pass


class DatasetTrainer(Trainer):
    """General Dataset Trainer with WandB integration."""

    def __init__(self, model, train_loader, val_loader, test_loader, configs, wb=False, logger=None, verbose=False):
        """
        Initialize the trainer.
        :param model: PyTorch model.
        :param train_loader: DataLoader for training set.
        :param val_loader: DataLoader for validation set.
        :param test_loader: DataLoader for testing set.
        :param configs: Configuration dictionary with hyperparameters.
        :param wb: Boolean to enable WandB logging (default False).
        :param logger: Logger instance for logging.
        :param verbose: Enable verbose logging (batch-level metrics).
        """
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.configs = configs
        self.wb = wb
        self.logger = logger if logger is not None else get_logger()
        self.verbose = verbose

        # Device setup
        self.device = torch.device(configs["device"])
        self.model = model.to(self.device)

        # Hyperparameters
        self.batch_size = configs["batch_size"]
        self.learning_rate = configs["lr"]
        self.min_lr = configs["min_lr"]
        self.weight_decay = configs["weight_decay"]
        self.optimizer_choice = configs["optimizer"]
        self.scheduler_choice = configs["lr_scheduler"]
        self.max_epochs = configs["max_epoch_num"]
        self.best_val_acc = 0.0

        # Optimizer, Scheduler, and Loss
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.criterion = self._initialize_loss().to(self.device)

        # Checkpoint and Logging
        self.start_time = datetime.datetime.now()
        self.checkpoint_path = configs.get("checkpoint_path", "best_model.pth")
        self.early_stopping_patience = configs.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0
        self.csv_log_path = configs.get("csv_log_path", "training_log.csv")
        self._initialize_csv_log()

        # WandB setup
        if self.wb:
            self._initialize_wandb()
        
        # Track learning rate changes
        self.previous_lr = self.learning_rate

    def _initialize_csv_log(self):
        # Open CSV log file and write header
        with open(self.csv_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate"
            ])

    def _initialize_optimizer(self):
        # phân nhóm tham số
        mha_params, backbone_params, no_decay = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'mha_block' in n or 'attn2' in n:
                mha_params.append(p)
            elif p.ndim == 1 or n.endswith('.bias') or 'norm' in n.lower():
                no_decay.append(p)
            else:
                backbone_params.append(p)

        groups = [
            {'params': backbone_params, 'lr': self.learning_rate},
            {'params': mha_params, 'lr': self.learning_rate * 0.1},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        if self.optimizer_choice == "SGD":
            return torch.optim.SGD(groups,
                                   lr=self.learning_rate,
                                   momentum=0.9,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_choice == "AdamW":
            return torch.optim.AdamW(
                groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        if self.optimizer_choice == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_choice == "RAdam":
            return torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            print("No learning rate scheduler selected.")
            return None




    def _initialize_scheduler(self):
        sched = self.scheduler_choice
        if sched == "CosineWarmup":
            warmup = self.configs.get("warmup_epochs", 10)
            max_ep = self.max_epochs
            min_lr = self.min_lr

            def lr_lambda(epoch):
                # linear warm-up
                if epoch < warmup:
                    return float(epoch + 1) / warmup
                # cosine decay to min_lr/base_lr
                progress = (epoch - warmup) / (max_ep - warmup)
                return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

            return LambdaLR(self.optimizer, lr_lambda)
        elif sched == "ReduceLROnPlateau":
            return ReduceLROnPlateau(self.optimizer, patience=5,
                                     factor=0.1, min_lr=self.min_lr)
        elif sched == "StepLR":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif sched == "CosineAnnealingLR":
            return CosineAnnealingLR(self.optimizer, T_max=10,
                                     eta_min=self.min_lr)
        elif sched == "CosineAnnealingWarmRestarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2,
                eta_min=self.min_lr
            )
        else:
            print("No learning rate scheduler selected.")

    def _initialize_loss(self):
        """Initialize loss function based on configuration."""
        loss_type = self.configs.get("loss_type", "cross_entropy")
        
        if loss_type == "focal":
            alpha = self.configs.get("focal_alpha", 0.25)
            gamma = self.configs.get("focal_gamma", 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            self.logger.warning(f"Unknown loss type: {loss_type}, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()

    def _initialize_wandb(self):
        """Initialize WandB."""
        api_key = self.configs.get("wandb_api_key", "")
        if not api_key or len(api_key) != 40:
            self.logger.warning("Invalid WandB API key, disabling WandB logging")
            self.wb = False
            return
        wandb.login(key=api_key)
        wandb.init(
            project=self.configs["project_name"],
            name=self.configs.get("run_name", f"run_{datetime.datetime.now()}"),
            config=self.configs,
        )
        wandb.watch(self.model, log="all", log_freq=10)
        self.logger.info("WandB initialized")

    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Check for learning rate changes
        current_lr = self.optimizer.param_groups[0]['lr']
        if abs(current_lr - self.previous_lr) > 1e-8:
            self.logger.log_lr_change(self.previous_lr, current_lr, "scheduler")
            self.previous_lr = current_lr
            
        # Show waiting message for first epoch
        if epoch == 1:
            print("Loading first batch (may take 30-60 seconds on CPU)...")
            print("Tip: Use GPU or reduce --batch_size for faster training")
            first_batch_start = time.time()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.max_epochs} [Training]",
            leave=False,
            ncols=100,
            disable=self.verbose,  # Disable progress bar in verbose mode
            dynamic_ncols=False,
            mininterval=1.0,
            maxinterval=10.0
        )
        
        for batch_idx, (images, labels) in pbar:
            # Show message for first batch
            if epoch == 1 and batch_idx == 0:
                if 'first_batch_start' in locals():
                    load_time = time.time() - first_batch_start
                    print(f"First batch loaded in {load_time:.1f}s! Training started...")
                else:
                    print("First batch loaded successfully! Training started...")
            
            # Simple progress indicator for verbose mode
            if self.verbose and batch_idx > 0 and (batch_idx % 100 == 0 or batch_idx == len(self.train_loader) - 1):
                progress = (batch_idx + 1) / len(self.train_loader) * 100
                print(f"  Training progress: {batch_idx+1}/{len(self.train_loader)} ({progress:.1f}%)")
                
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log batch-level metrics
            batch_loss = loss.item()
            batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar with current metrics
            if not self.verbose:
                # Only update postfix in normal mode
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'Acc': f'{batch_acc:.1f}%',
                    'LR': f'{current_lr:.2e}'
                })
            # Verbose mode: No postfix updates to avoid spam
            
            self.logger.log_batch(epoch, batch_idx, len(self.train_loader), 
                                batch_loss, batch_acc, current_lr, phase="train", verbose=self.verbose)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        # Print summary for verbose mode
        if self.verbose:
            print(f"  Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {current_lr:.2e}")
        else:
            # Clear progress bar line for normal mode
            pass

        # Log to WandB
        if self.wb:
            wandb.log({"Train Loss": avg_loss, "Train Accuracy": accuracy, "Learning Rate": current_lr})

        return avg_loss, accuracy

    def validate_one_epoch(self, epoch):
        """Validate the model for one epoch."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_pbar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc=f"Epoch {epoch}/{self.max_epochs} [Validation]",
                leave=False,
                ncols=100,
                disable=self.verbose,  # Disable progress bar in verbose mode
                dynamic_ncols=False,
                mininterval=1.0,
                maxinterval=10.0
            )
            
            for batch_idx, (images, labels) in val_pbar:
                # Simple progress indicator for verbose mode
                if self.verbose and batch_idx > 0 and (batch_idx % 50 == 0 or batch_idx == len(self.val_loader) - 1):
                    progress = (batch_idx + 1) / len(self.val_loader) * 100
                    print(f"  Validation progress: {batch_idx+1}/{len(self.val_loader)} ({progress:.1f}%)")
                    
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update validation progress bar
                batch_loss = loss.item()
                batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update validation progress bar
                if not self.verbose:
                    # Only update postfix in normal mode
                    val_pbar.set_postfix({
                        'Loss': f'{batch_loss:.4f}',
                        'Acc': f'{batch_acc:.1f}%'
                    })
                # Verbose mode: No postfix updates to avoid spam
                
                self.logger.log_batch(epoch, batch_idx, len(self.val_loader), 
                                    batch_loss, batch_acc, current_lr, phase="val", verbose=self.verbose)

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        # Print summary for verbose mode
        if self.verbose:
            print(f"  Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Log to WandB
        if self.wb:
            wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy})

        return avg_loss, accuracy

    def test_model(self):
        """Evaluate the model on the test set."""
        self.logger.info("Starting model evaluation on test set...")
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            test_pbar = tqdm(
                self.test_loader, 
                total=len(self.test_loader), 
                desc="Testing Final Model",
                ncols=100
            )
            
            for images, labels in test_pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update test progress bar
                current_acc = 100.0 * correct / total
                test_pbar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})

        accuracy = 100.0 * correct / total

        # Log to WandB
        if self.wb:
            wandb.log({"Test Accuracy": accuracy})

        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def train(self):
        """Main training loop."""
        train_hist = {"loss": [], "accuracy": []}
        val_hist = {"loss": [], "accuracy": []}
        
        # Log training start
        self.logger.log_training_start(self.max_epochs, self.optimizer, self.scheduler)
        
        # Show initialization progress
        print("Initializing training components...")
        init_pbar = tqdm(total=4, desc="Training Setup", ncols=80)
        
        init_pbar.set_description("Setting up data loaders...")
        init_pbar.update(1)
        
        init_pbar.set_description("Preparing model for training...")
        init_pbar.update(1)
        
        init_pbar.set_description("Initializing optimizer & scheduler...")
        init_pbar.update(1)
        
        init_pbar.set_description("Starting training loop...")
        init_pbar.update(1)
        init_pbar.close()
        
        print("Setup complete! Starting training...\n")
        training_start_time = time.time()

        early_stopper = EarlyStopping(patience=self.early_stopping_patience)

        # Training loop with overall progress
        epoch_pbar = tqdm(
            range(1, self.max_epochs + 1),
            desc="Training Progress",
            ncols=120,
            position=0
        )
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # Log epoch start
            self.logger.log_epoch_start(epoch, self.max_epochs)
            
            # Show data loading preparation
            if epoch == 1:
                print("Preparing first batch (this may take a moment)...")
                data_prep_pbar = tqdm(total=3, desc="Data Preparation", ncols=80, leave=False)
                
                data_prep_pbar.set_description("Loading first batch...")
                data_prep_pbar.update(1)
                
                data_prep_pbar.set_description("Applying transforms...")
                data_prep_pbar.update(1)
                
                data_prep_pbar.set_description("Moving to device...")
                data_prep_pbar.update(1)
                data_prep_pbar.close()
            
            # Train and validate
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate_one_epoch(epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update overall progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.1f}%',
                'Val Loss': f'{val_loss:.4f}', 
                'Val Acc': f'{val_acc:.1f}%',
                'Best': f'{self.best_val_acc:.1f}%'
            })
            
            # Append training history
            train_hist["loss"].append(train_loss)
            train_hist["accuracy"].append(train_acc)
            val_hist["loss"].append(val_loss)
            val_hist["accuracy"].append(val_acc)

            # Scheduler step
            old_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Log learning rate change if it occurred
            if abs(old_lr - new_lr) > 1e-8:
                self.logger.log_lr_change(old_lr, new_lr, "scheduler")
            
            # Log epoch summary
            self.logger.log_epoch_end(epoch, train_loss, train_acc, val_loss, val_acc, new_lr, epoch_time)
            
            # Write to CSV log
            with open(self.csv_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    new_lr
                ])

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.logger.log_checkpoint_save(epoch, self.checkpoint_path, val_acc)
                
            # Early stopping check
            early_stopper(val_acc)
            if early_stopper.early_stop:
                self.logger.log_early_stopping(epoch, self.early_stopping_patience)
                break

        # Calculate total training time
        total_training_time = time.time() - training_start_time
        
        # Log training completion
        self.logger.log_training_end(self.best_val_acc, total_training_time)

        # Log final summary to WandB
        if self.wb:
            wandb.log({
                "Best Validation Accuracy": self.best_val_acc,
                "Final Training Loss": train_hist["loss"][-1],
                "Final Training Accuracy": train_hist["accuracy"][-1],
                "Final Validation Loss": val_hist["loss"][-1],
                "Final Validation Accuracy": val_hist["accuracy"][-1],
                "Total Training Time": total_training_time
            })
