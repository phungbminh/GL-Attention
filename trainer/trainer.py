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
import tqdm
from torch.utils.data import DataLoader
import wandb  # Import wandb for logging
import csv
import math
from utils.logger import get_logger

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
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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
                                     factor=0.1, min_lr=self.min_lr, verbose=True)
        elif sched == "StepLR":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif sched == "CosineAnnealingLR":
            return CosineAnnealingLR(self.optimizer, T_max=10,
                                     eta_min=self.min_lr)
        elif sched == "CosineAnnealingWarmRestarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2,
                eta_min=self.min_lr, verbose=True
            )
        else:
            print("No learning rate scheduler selected.")
            return None

    def _initialize_wandb(self):
        """Initialize WandB."""
        wandb.login(key=self.configs["wandb_api_key"])
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

        for batch_idx, (images, labels) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}/{self.max_epochs} [Training]",
                leave=False):
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
            
            self.logger.log_batch(epoch, batch_idx, len(self.train_loader), 
                                batch_loss, batch_acc, current_lr, phase="train", verbose=self.verbose)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

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
            for batch_idx, (images, labels) in tqdm.tqdm(
                    enumerate(self.val_loader),
                    total=len(self.val_loader),
                    desc=f"Epoch {epoch}/{self.max_epochs} [Validation]",
                    leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Log batch-level metrics for validation
                batch_loss = loss.item()
                batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.logger.log_batch(epoch, batch_idx, len(self.val_loader), 
                                    batch_loss, batch_acc, current_lr, phase="val", verbose=self.verbose)

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

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
            for images, labels in tqdm.tqdm(
                    self.test_loader, total=len(self.test_loader), desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

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
        training_start_time = time.time()

        early_stopper = EarlyStopping(patience=self.early_stopping_patience)

        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()
            
            # Log epoch start
            self.logger.log_epoch_start(epoch, self.max_epochs)
            
            # Train and validate
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate_one_epoch(epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
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
