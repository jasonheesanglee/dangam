from setuptools import setup, find_packages

setup(
    name='dangam',
    version='0.1.0',
    packages=find_packages(),
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jason Heesang Lee',
    author_email='jason.heesang.lee96@gmail.com',
    url='https://github.com/jasonheesanglee/dangam',
    install_requires=[
        # List of dependencies,
    ],
    classifiers=[
        # Development status, intended audience, license, etc.
        # See: https://pypi.org/classifiers/
    ]
)
