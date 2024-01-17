from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='yume',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            
        ],
    },
    author='Zai',
    author_email='zaiyellyintaung@gmail.com', 
    description='Diffusion model implementation with PyTorch',
    long_description='Detailed description of your project',
    url='https://github.com/zaibutcooler/yume',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
