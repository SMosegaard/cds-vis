# Activate the virtual envoriment
source ./env/bin/activate 

# Run the code
python src/src.py "$@"

# Close the virtual envoriment
deactivate 
