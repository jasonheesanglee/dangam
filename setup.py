from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DanGam',
    version='0.0.129',
    packages=find_packages(),
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jason Heesang Lee',
    author_email='jason.heesang.lee96@gmail.com',
    url='https://github.com/jasonheesanglee/dangam',
    install_requires=requirements,
    classifiers=[
        # Choose the development status of your package
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',

        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',

        'Operating System :: OS Independent',  # Use specific OS classifiers if your package is OS-dependent

    ]
)
