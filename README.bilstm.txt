To train the model:
python bilstmTrain.py [repr] [train_file] [model_file] [dev_file]

For example:
python bilstmTrain.py d pos/train model_pos_d.pt pos/dev

To generate predictions:
python bilstmPredict.py [repr] [model_file] [test_file] > test4.[pos/ner]

Example:
python bilstmPredict.py d model_pos_d.pt pos/test > test4.pos
