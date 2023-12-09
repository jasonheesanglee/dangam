from setuptools import setup, find_packages

setup(
    name='DanGam',
    version='0.1.2',
    packages=find_packages(),
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jason Heesang Lee',
    author_email='jason.heesang.lee96@gmail.com',
    url='https://github.com/jasonheesanglee/dangam',
    install_requires=[
        'torch>=1.10.0',
        'numpy>=1.21.0',
        'tqdm>=4.62.0',
        'transformers>=4.10.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'regex>=2021.8.0'
    ],
    classifiers=[
        # Choose the development status of your package
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',

        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',

        'Programming Language :: Python :: 3.10',

        'Operating System :: OS Independent',  # Use specific OS classifiers if your package is OS-dependent

    ]
)
