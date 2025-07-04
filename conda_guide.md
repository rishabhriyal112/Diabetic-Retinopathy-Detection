Here is a proper setup guide for Conda locally:

**Step 1: Install Miniconda**

If you haven't already, download and install Miniconda from the official website: <https://docs.conda.io/en/latest/miniconda.html>

**Step 2: Create a New Environment**

Open a terminal or command prompt and run the following command to create a new environment:
```bash
conda create --name diabetic_retinopathy
```
This will create a new environment named `diabetic_retinopathy`.

**Step 3: Activate the Environment**

To activate the environment, run the following command:
```bash
conda activate diabetic_retinopathy
```
You should see the environment name printed in your terminal or command prompt.

**Step 4: Install Required Packages**

To install the required packages, run the following command:
```bash
conda install -c conda-forge tensorflow numpy pandas matplotlib scikit-learn opencv
```
This will install the required packages, including TensorFlow, NumPy, Pandas, Matplotlib, Scikit-learn, and OpenCV.

**Step 5: Install Keras**

To install Keras, run the following command:
```bash
conda install -c conda-forge keras
```
**Step 6: Verify the Installation**

To verify that the installation was successful, run the following command:
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```
This should print the version of TensorFlow installed.

**Step 7: Deactivate the Environment**

To deactivate the environment, run the following command:
```bash
conda deactivate
```
You can now use the `diabetic_retinopathy` environment to run your project.

**Tips and Variations**

* To update the environment, run `conda update --all` while the environment is activated.
* To list all installed packages, run `conda list` while the environment is activated.
* To remove a package, run `conda remove <package_name>` while the environment is activated.
* To create a new environment with a different Python version, run `conda create --name <env_name> python=<python_version>`.