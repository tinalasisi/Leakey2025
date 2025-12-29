# OpenApePose
Jan Zimmermann (University of Minnesota)

---

OpenApePose is an annotated image dataset for training models on ape and monkey computer vision pose estimation

## Description of the data and file structure

Unpack all archives.

Dependencies: OpenMMLab; MMdetect, MMpose

/annotations # contains annotations of all images
-> annotations are in standard COCO format
/code # code for training the supplied deep learning model
-> animal_oap_dataset.py is the general dataloader for the OpenApePose dataset
-> hrnet_w48_oap_256x192_full.py is the general model structure
-> TopDownOAPDataset.py defines the dictionary keypoints
-> __init__.py is the namespace
/images # all images used in training
/models # final model for benchmarking
-> hrnet_w48_oap_256x192_full.pth final training save point of the model
