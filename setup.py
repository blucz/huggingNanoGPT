from setuptools import setup, find_packages

setup(
    name='hugging_nanogpt',
    version='0.1',
    description='🤗transformers style model that is compatible with nanoGPT checkpoints.',
    author='Brian Luczkiewicz',
    author_email='brian@blucz.com',
    url='https://github.com/blucz/huggingNanoGPT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18',
        'pandas>=1.1',
        'scikit-learn>=0.24',
    ],
    python_requires=">=3.7",
    license="Apache 2.0",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
