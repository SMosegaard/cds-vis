# Create virtual envoriment called env
python -m venv env

# Activate the virtual env
source ./env/bin/activate

# Install opencv 
sudo apt-get update
sudo apt-get install -y python3-opencv

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Close the envoriment
deactivate
