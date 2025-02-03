from setuptools import setup, find_packages

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
)