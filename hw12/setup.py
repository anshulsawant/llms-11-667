import setuptools
import os

def read_requirements(file_path="requirements.txt"):
    """Reads requirements from a file, ignoring comments and empty lines."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. No requirements will be installed.")
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()
        # Filter out comments and empty lines
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]
    return requirements

# Read the contents of your README file (optional, assumes README.md exists)
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Supervised Fine-Tuning (SFT) project using Hugging Face Transformers."

# Define the package name (should match the directory name inside src/)
package_name = "sft_project" # Or whatever you name your package directory inside src/

setuptools.setup(
    name=package_name, # Use the defined package name
    version="0.1.0", # Initial version
    author="Anshul Sawant", # Replace with your name
    author_email="anshul.sawant@gmail.com", # Replace with your email
    description="A project for supervised fine-tuning of language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anshulsawant/llms-11-667",
    package_dir={"": "src"}, # Tell setuptools that packages are under src/
    packages=setuptools.find_packages(where="src"), # Find packages in src/
    python_requires='>=3.9', # Specify your minimum Python version requirement
    install_requires=read_requirements("requirements.txt"), # Read dependencies
)
