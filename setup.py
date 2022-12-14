from setuptools import setup, find_packages

setup(
    name = 'vision_transformers',
    packages = find_packages(),
    version = '0.0.2',
    license = 'MIT',
    description = 'Vision Transformers (ViT)',
    long_description_content_type = 'text/markdown',
    author = 'Sovit Rath',
    author_email = 'sovitrath5@gmail.com',
    url = 'https://github.com/sovit-123/vision_transformers',
    keywords = [
        'vision',
        'neural attention',
        'deep learning'
    ],
    install_requires = [
        'torch>=1.10',
        'torchvision'
    ],
    setup_requires = [],
    tests_require = [],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)