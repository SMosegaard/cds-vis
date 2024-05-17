# Activate the envoriment
source ./env/bin/activate 

# Run the code
python src/document_classifier.py "$@"

# Close the envoriment
deactivate