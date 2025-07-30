import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
from pathlib import Path


class ProjectLogger:
    """Enhanced logger for the entire project with different log levels and formatting."""
    
    def __init__(self, log_dir="logs", project_name="GL-Attention", console_level=logging.INFO, file_level=logging.DEBUG):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Training metrics handler (separate file)
        metrics_file = self.log_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.metrics_handler = logging.FileHandler(metrics_file)
        self.metrics_handler.setLevel(logging.INFO)
        metrics_formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.metrics_handler.setFormatter(metrics_formatter)
        
        # Create separate logger for metrics
        self.metrics_logger = logging.getLogger(f"{project_name}_metrics")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        self.metrics_logger.addHandler(self.metrics_handler)
        
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_config(self, config_dict, title="Configuration"):
        """Log configuration dictionary"""
        self.info(f"{title}:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")
    
    def log_model_info(self, model, title="Model Information"):
        """Log model architecture and parameter count"""
        self.info(f"{title}:")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"  Total parameters: {total_params:,}")
        self.info(f"  Trainable parameters: {trainable_params:,}")
        self.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Model architecture (first few layers)
        self.debug(f"  Model architecture:\n{str(model)}")
    
    def log_dataset_info(self, train_size, val_size, test_size, num_classes):
        """Log dataset information"""
        self.info("Dataset Information:")
        self.info(f"  Train samples: {train_size:,}")
        self.info(f"  Validation samples: {val_size:,}")
        self.info(f"  Test samples: {test_size:,}")
        self.info(f"  Number of classes: {num_classes}")
        self.info(f"  Total samples: {train_size + val_size + test_size:,}")
    
    def log_epoch_start(self, epoch, total_epochs):
        """Log epoch start"""
        self.info(f"Starting Epoch {epoch}/{total_epochs}")
        self.info("-" * 50)
    
    def log_epoch_end(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Log epoch summary"""
        self.info(f"Epoch {epoch} Summary:")
        self.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        self.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        self.info(f"  Learning Rate: {lr:.6f}")
        self.info(f"  Epoch Time: {epoch_time:.2f}s")
        self.info("-" * 50)
        
        # Log to metrics file
        metrics_msg = f"EPOCH,{epoch},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},{lr:.6f},{epoch_time:.2f}"
        self.metrics_logger.info(metrics_msg)
    
    def log_batch(self, epoch, batch_idx, total_batches, loss, acc, lr, phase="train", verbose=False):
        """Log batch information"""
        # Always log to metrics file
        metrics_msg = f"BATCH,{epoch},{batch_idx},{phase},{loss:.4f},{acc:.2f},{lr:.6f}"
        self.metrics_logger.info(metrics_msg)
        
        # Console logging: completely disabled for verbose mode
        if verbose:
            return  # No console logging in verbose mode - progress bar handles this
            
        # Only log milestone batches (every 100 batches) in non-verbose mode
        if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            self.info(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) | "
                     f"Loss: {loss:.4f} | Acc: {acc:.2f}% | LR: {lr:.6f}")
    
    def log_training_start(self, total_epochs, optimizer, scheduler):
        """Log training start information"""
        self.info("=" * 60)
        self.info("TRAINING STARTED")
        self.info("=" * 60)
        self.info(f"Total epochs: {total_epochs}")
        self.info(f"Optimizer: {optimizer.__class__.__name__}")
        if scheduler:
            self.info(f"Scheduler: {scheduler.__class__.__name__}")
        self.info("")
    
    def log_training_end(self, best_acc, total_time):
        """Log training completion"""
        self.info("=" * 60)
        self.info("TRAINING COMPLETED")
        self.info("=" * 60)
        self.info(f"Best validation accuracy: {best_acc:.2f}%")
        self.info(f"Total training time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        self.info("")
    
    def log_early_stopping(self, epoch, patience):
        """Log early stopping"""
        self.warning(f"Early stopping triggered at epoch {epoch} (patience: {patience})")
    
    def log_checkpoint_save(self, epoch, path, metric_value, metric_name="accuracy"):
        """Log checkpoint saving"""
        self.info(f"New best {metric_name}: {metric_value:.2f}% at epoch {epoch}")
        self.info(f"Model saved to: {path}")
    
    def log_lr_change(self, old_lr, new_lr, reason="scheduler"):
        """Log learning rate changes"""
        self.info(f"Learning rate changed: {old_lr:.6f} -> {new_lr:.6f} ({reason})")
    
    def log_fusion_strategy(self, fusion_strategy, residual_strategy):
        """Log fusion strategy information"""
        self.info("Fusion Strategy Configuration:")
        self.info(f"  Fusion Strategy: {fusion_strategy}")
        self.info(f"  Residual Strategy: {residual_strategy}")
    
    def close(self):
        """Close all handlers"""
        for handler in self.logger.handlers:
            handler.close()
        for handler in self.metrics_logger.handlers:
            handler.close()


# Legacy Logger class for backward compatibility
class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, resume=False): 
        self.file = None
        self.resume = resume
        if os.path.isfile(fpath):
            if resume:
                self.file = open(fpath, 'a') 
            else:
                self.file = open(fpath, 'w')
        else:
            self.file = open(fpath, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
                self.file.write(target_str + '\n')
                self.file.flush()
        else:
            print(target_str)
            self.file.write(target_str + '\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


# Global logger instance
_global_logger = None

def get_logger():
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ProjectLogger()
    return _global_logger

def setup_logger(log_dir="logs", project_name="GL-Attention", console_level=logging.INFO, file_level=logging.DEBUG):
    """Setup global logger"""
    global _global_logger
    _global_logger = ProjectLogger(log_dir, project_name, console_level, file_level)
    return _global_logger