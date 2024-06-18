from setuptools import setup, find_packages

setup(
    name='deepcover',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'tensorflow',
        'keras',
        'matplotlib'
    ],
    description='A package to explain image classifications using Grad-CAM and responsibility calculation.',
    author='Souham Sengupta, Andrei Luca Rusu',
    author_email='senguptasouham@gmail.com',
    url='https://github.com/rickOS98/DeepCover',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)