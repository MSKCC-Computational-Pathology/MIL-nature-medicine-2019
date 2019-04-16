# MIL-nature-medicine-2019
This repository provides training and testing scripts for the article *Campanella et al. 2019*.

## Weakly-supervised tile level classifier

### MIL Input Data
Input data, whether for training, validation or testing, should be a dictionary stored to disk with `torch.save()` containing the following keys:
* `"slides"`: list of full paths to WSIs (e.g. `my/full/path/slide01.svs`). Size of the list equals the number of slides.
* `"grid"`: list of a list of tuple (x,y) coordinates. Size of the list is equal to the number of slides. Size of each sublist is equal to the number of tiles in each slide. An example grid list containing two slides, one with 3 tiles and one with 4 tiles:
```python
grid = [
        [(x1_1, y1_1),
	 (x1_2, y1_2),
	 (x1_3, y1_3)],
	[(x2_1, y2_1),
	 (x2_2, y2_2),
	 (x2_3, y2_3),
	 (x2_4, y2_4)],
]
```
* `"targets"`: list of slide level class (0: benign slide, 1: tumor slide). Size of the list equals the number of slides.
* `"mult"`: scale factor (float) for achieving resolutions different than the ones saved in the WSI pyramid file. Usually `1.` for no scaling.
* `"level"`: WSI pyramid level (integer) from which to read the tiles. Usually `0` for the highest resolution.

### MIL Training
To train a model, use script `MIL_train.py`. Run `python MIL_train.py -h` to get help regarding input parameters.
Script outputs:
* **convergence.csv**: *.csv* file containing training loss and validation error metrics.
* **checkpoint_best.pth**: file containing the weights of the best model on the validation set. This file can be used with the `MIL_test.py` script to run the model on a test set. In addition, this file can be used to generate the embedding needed to train the RNN aggregator.

### MIL Testing
To run a model on a test set, use script `MIL_test.py`. Run `python MIL_test.py -h` to get help regarding input parameters.
Script outputs:
* **predictions.csv**: *.csv* file with slide name, slide target, model prediction and tumor probability entries for each slide in the test data. This file can be used to generate confusion matrix, ROC curve and AUC.

## RNN Aggregator

### RNN Input Data
Input data, whether for training, validation or testing, should be a dictionary stored to disk with `torch.save()` containing the following keys:
* `"slides"`: list of full paths to WSIs (e.g. `my/full/path/slide01.svs`). Size of the list equals the number of slides.
* `"grid"`: list of a list of tuple (x,y) coordinates. Size of the list is equal to the number of slides. Size of each sublist is equal to the number of maximum number of recurrent steps (we used 10). Each sublist is in decreasing order of tumor probability.
* `"targets"`: list of slide level class (0: benign slide, 1: tumor slide). Size of the list equals the number of slides.
* `"mult"`: scale factor (float) for achieving resolutions different than the ones saved in the WSI pyramid file. Usually `1.` for no scaling.
* `"level"`: WSI pyramid level (integer) from which to read the tiles. Usually `0` for the highest resolution.

### RNN Training
To train the RNN aggregator model, use script `RNN_train.py`. Run `python RNN_train.py -h` to get help regarding input parameters. You will need to have a trained embedder using the script `MIL_train.py`.
Script outputs:
* **convergence.csv**: *.csv* file containing training loss and validation error metrics.
* **rnn_checkpoint_best.pth**: file containing the weights of the best model on the validation set. This file can be used with the `RNN_test.py` script.

### RNN Testing
To run a model on a test set, use script `RNN_test.py`. Run `python RNN_test.py -h` to get help regarding input parameters.
Script outputs:
* **predictions.csv**: *.csv* file with slide name, slide target, model prediction and tumor probability entries for each slide in the test data. This file can be used to generate confusion matrix, ROC curve and AUC.
