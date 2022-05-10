madhappy

	emotion based filter application utilizing 3 emotions: happy, sad, surprise, and one default: neutral
	We deleted other training checkpoints except our highest accuracy - 87.08 - found in /data/checkpoints
	
File structure:

	filter_app.py - facial recognition and application of filters
	cnn.py -  CNN architecture
	main.py - train model methods
	preprocess.py - preprocessing data (FER2013 dataset)
	regularization.py - data augmentation

TO RUN THIS PROJECT:

	change path to be your own username in filter_app.py - line 11
	ex: model.load_weights('/Users/<YOUR USERNAME HERE>/cs1430/madhappy/data/checkpoints/your.weights.e017-acc0.8708.h5')

	run filter_app.py from madhappy directory - this will automatically load our best weights
	ex: python filter_app.py

	test your different emotions to see the various filters!
