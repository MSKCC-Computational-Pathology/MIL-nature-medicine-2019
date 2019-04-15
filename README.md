# MIL-nature-medicine-2019
This repository blah

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

### MIL Testing

## RNN Aggregator

### RNN Input Data

### RNN Training

### RNN Testing
