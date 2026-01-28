"""
Download CIFAR-10 and CIFAR-100 datasets
"""
from torchvision import datasets, transforms
import os


def download_cifar10():
    """Download CIFAR-10 dataset"""
    print("Downloading CIFAR-10...")

    # Create data directory if it doesn't exist
    os.makedirs('./data/cifar-10', exist_ok=True)

    # Download training set
    datasets.CIFAR10(root='./data/cifar-10', train=True, download=True)

    # Download test set
    datasets.CIFAR10(root='./data/cifar-10', train=False, download=True)

    print("✓ CIFAR-10 downloaded successfully!")


def download_cifar100():
    """Download CIFAR-100 dataset"""
    print("Downloading CIFAR-100...")

    # Create data directory if it doesn't exist
    os.makedirs('./data/cifar-100', exist_ok=True)

    # Download training set
    datasets.CIFAR100(root='./data/cifar-100', train=True, download=True)

    # Download test set
    datasets.CIFAR100(root='./data/cifar-100', train=False, download=True)

    print("✓ CIFAR-100 downloaded successfully!")


def verify_datasets():
    """Verify datasets are downloaded correctly"""
    print("\nVerifying datasets...")

    # Load CIFAR-10
    cifar10_train = datasets.CIFAR10(root='./data/cifar-10', train=True, download=False)
    cifar10_test = datasets.CIFAR10(root='./data/cifar-10', train=False, download=False)
    print(f"✓ CIFAR-10 train: {len(cifar10_train)} samples")
    print(f"✓ CIFAR-10 test: {len(cifar10_test)} samples")

    # Load CIFAR-100
    cifar100_train = datasets.CIFAR100(root='./data/cifar-100', train=True, download=False)
    cifar100_test = datasets.CIFAR100(root='./data/cifar-100', train=False, download=False)
    print(f"✓ CIFAR-100 train: {len(cifar100_train)} samples")
    print(f"✓ CIFAR-100 test: {len(cifar100_test)} samples")


if __name__ == '__main__':
    print("=" * 60)
    print("Downloading Datasets")
    print("=" * 60)

    download_cifar10()
    download_cifar100()
    verify_datasets()

    print("=" * 60)
    print("✓ All datasets ready!")
    print("=" * 60)
