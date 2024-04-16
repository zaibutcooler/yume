from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="yume",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Zai",
    author_email="zaiyellyintaung@gmail.com",
    description="LLM trained with Animanga dataset",
    long_description="Inspired by Andrej Karpathy trained with japanese animanga dataset",
    url="https://github.com/zaibutcooler/yume",
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
)
