# setup.py
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()
    # Filter out comments and empty lines
    required = [line for line in required if line and not line.startswith('#')]
    # Exclude pytest from core install_requires, put in extras_require['dev']
    dev_requires = [req for req in required if 'pytest' in req]
    install_requires = [req for req in required if 'pytest' not in req]


setup(
    name='ppo_rl_tutorial',
    version='0.1.0',
    description='A simplified PPO RL tutorial for LLMs based on TinyZero exercises.',
    author='Anshul Sawant', # Change as needed
    packages=find_packages(where='src'), # Look for packages in src/
    package_dir={'': 'src'}, # Tell setuptools that packages are under src
    install_requires=install_requires, # Install core dependencies
    extras_require={
        'dev': dev_requires + [
            # Add other development dependencies here if needed
            # e.g., 'black', 'flake8'
        ]
    },
    python_requires='>=3.9', # Specify compatible Python versions
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License', # Adjust if needed
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        # Optional: Define command-line entry points if needed
        # 'console_scripts': [
        #     'run_ppo_training=ppo_trainer:main_cli', # Example if you wrap main logic
        # ],
    },
)
