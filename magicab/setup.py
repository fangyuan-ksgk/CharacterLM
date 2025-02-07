from setuptools import setup, find_packages
from setuptools.dist import Distribution
from setuptools_rust import RustExtension, Binding

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name="magicab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "matplotlib",
        "transformers",
        "datasets",
        "tiktoken",
        "wandb",
        "tqdm",
        "accelerate",
        "SoMaJo"
    ],
    author="Fangyuan Yu",
    author_email="fangyuan.yu18@gmail.com",
    description="A package for building LLM with trainable tokenizer",
    python_requires=">=3.6",
    setup_requires=["setuptools-rust>=1.5.2"],
    distclass=BinaryDistribution,
    rust_extensions=[
        RustExtension(
            "rust_tokenizer.rust_tokenizer",
            "rust_tokenizer/Cargo.toml",
            binding=Binding.PyO3
        )
    ]
)