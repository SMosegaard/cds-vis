# Create virtual envoriment called env
python -m venv env

# Activate the virtual envoriment
source ./env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Close the virtual envoriment
deactivate