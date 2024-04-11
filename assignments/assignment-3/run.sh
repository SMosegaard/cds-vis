# Activate the envoriment
source ./env/bin/activate 

# Run the pretrained CNN
python src/TransferLearning.py "$@"

# Run the pretrained CNN with batch normalization
python src/TransferLearning_BatchNorm.py "$@"

# Run the pretrained CNN with data augmentation
python src/TransferLearning_DatAug.py "$@"

# Close the envoriment
deactivate