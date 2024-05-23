# Activate the virtual envoriment
source ./env/bin/activate 

# Run the code
python src/document_classifier.py "$@"

# Close the virtual envoriment
deactivate