# Create virtual envoriment called env
python -m venv env

# Go to file called avtivate - run it and activate the virt env
source ./env/bin/activate   

# Install requirements and all the packages specified in the requirements.txt file
pip install --upgrade pip
pip install -r requirements.txt 

# Close the envoriment
deactivate      