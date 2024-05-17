# Create a virtual envoriment called env
python -m venv env

# Activate the virtual envoriment
source ./env/bin/activate

# Install opencv
sudo apt-get update
sudo apt-get install -y python3-opencv

# Install requirements
pip install --upgrade pip
pip install -r requirements_explainer.txt

# Run the code
python src/explainer.py "$@"

# Close the virtual envoriment
deactivate