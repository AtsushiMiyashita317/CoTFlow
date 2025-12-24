from setuptools import setup, find_packages

setup(
    name="cotflow",
    version="0.1.0",
    description="Conditional Flow-based generative models with autoencoder for CelebA and 3D Shapes datasets.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "tqdm",
        "gdown",
        "wandb",
        "hydra-core",
        "omegaconf",
        "h5py",
        "normflows @ git+https://github.com/AtsushiMiyashita317/normalizing-flows.git",
    ],
    entry_points={
        "console_scripts": [
            "train_autoencoder=cotflow.bin.train_autoencoder:main",
            "train_predictor=cotflow.bin.train_predictor:main",
            "train_cotflow=cotflow.bin.train_cotflow:main",
            "train_cotglow=cotflow.bin.train_cotglow:main",
            "dump_features=cotflow.bin.dump_features:main",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
)
