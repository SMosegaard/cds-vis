# Activate the virtual envoriment
source ./env/bin/activate 

# Run the code
python src/image_search.py "$@"

# Close the virtual envoriment
deactivate 
