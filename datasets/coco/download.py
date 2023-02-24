import fiftyone.zoo as foz
import fiftyone as fo


dataset = foz.load_zoo_dataset(
    'coco-2017',
    split = "train",
    label_types = ['detections'],
    classes = ['car'],
    max_samples = 50, 
    only_matching = True,
    dataset_dir = '/Users/marvin/Documents/codes/python-algorithm/data/coco/',
)
session = fo.launch_app(dataset)
session.wait()
