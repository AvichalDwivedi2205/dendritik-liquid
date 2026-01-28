"""
Train Baseline Vision Transformer on CIFAR-10

This establishes the baseline performance that DendriticLiquid-ViT will be compared against.

Usage:
    python experiments/train_baseline.py --epochs 50 --batch_size 128
"""
import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vit_base import create_vit_small, create_vit_tiny, create_vit_base
from src.data.cifar_dataset import get_cifar10_dataloaders


class ViTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Vision Transformer

    Handles training, validation, and optimization
    """
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_epochs=5,
        max_epochs=50
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.criterion = torch.nn.CrossEntropyLoss()

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        """
        Configure AdamW optimizer with cosine annealing and warmup
        """
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def main(args):
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Create model
    if args.model_size == 'tiny':
        model = create_vit_tiny(num_classes=10)
    elif args.model_size == 'small':
        model = create_vit_small(num_classes=10)
    elif args.model_size == 'base':
        model = create_vit_base(num_classes=10)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    print(f"\n{'='*70}")
    print(f"Training Vision Transformer ({args.model_size.upper()})")
    print(f"{'='*70}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create Lightning module
    lit_model = ViTLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs
    )

    # Data loaders
    train_loader, val_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/baseline_{args.model_size}',
        filename='vit-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Logger
    if args.use_wandb:
        logger = WandbLogger(
            project='dendriticliquid-vit',
            name=f'baseline-vit-{args.model_size}-seed{args.seed}',
            save_dir='logs'
        )
    else:
        logger = None

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        precision='16-mixed' if args.use_amp else 32,
        deterministic=True,
        log_every_n_steps=10
    )

    print(f"\n{'='*70}")
    print("Starting Training...")
    print(f"{'='*70}\n")

    # Train
    trainer.fit(lit_model, train_loader, val_loader)

    # Test on validation set (final evaluation)
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")
    result = trainer.validate(lit_model, val_loader)

    print(f"\n✓ Training completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Final validation accuracy: {result[0]['val_acc']*100:.2f}%")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Baseline ViT on CIFAR-10')

    # Model
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'base'],
                        help='Model size')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')

    args = parser.parse_args()

    # Run training
    main(args)
