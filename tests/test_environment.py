"""
Test environment setup - verify GPU, PyTorch, and dependencies
"""
import torch
import sys


def test_python_version():
    """Verify Python version"""
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"
    print(f"✓ Python version: {sys.version.split()[0]}")


def test_pytorch_installed():
    """Verify PyTorch is installed"""
    print(f"✓ PyTorch version: {torch.__version__}")
    assert torch.__version__, "PyTorch not installed!"


def test_cuda_available():
    """Verify CUDA is available"""
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA not available - will use CPU (training will be slow)")


def test_basic_tensor_operations():
    """Test basic tensor operations"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create tensor
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)

    # Test operations
    z = x @ y
    assert z.shape == (10, 10), "Matrix multiplication failed"

    print(f"✓ Tensor operations working on {device}")


def test_imports():
    """Test that all required packages can be imported"""
    packages = [
        'torch',
        'torchvision',
        'pytorch_lightning',
        'timm',
        'wandb',
        'einops',
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn',
        'pytest',
        'fvcore',
        'yaml'
    ]

    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError:
            failed.append(package)
            print(f"✗ {package} import failed")

    assert len(failed) == 0, f"Failed to import: {', '.join(failed)}"


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Environment Setup")
    print("=" * 60)

    test_python_version()
    test_pytorch_installed()
    test_cuda_available()
    test_basic_tensor_operations()
    test_imports()

    print("=" * 60)
    print("✓ All environment tests passed!")
    print("=" * 60)
