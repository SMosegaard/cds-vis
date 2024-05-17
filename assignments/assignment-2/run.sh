# Activate the virtual envoriment
source ./env/bin/activate 

# Run the LR classifier
python src/LR_classifier.py "$@"

# Run the NN classifier
python src/NN_classifier.py "$@"

# Close the virtual envoriment
deactivate