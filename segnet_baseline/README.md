# SegNet baseline

This is a compact PyTorch reproduction of the SegNet idea from Badrinarayanan et al., using VGG-style convolution blocks and max-pooling indices retained for `MaxUnpool2d` decoding. `train.py` currently provides a deterministic end-to-end smoke test on the repository's synthetic road/grass generator.

For a data-faithful experiment, replace the dataset in `train.py` with CamVid: map the `Road` annotation label to class 1 and all other labels to class 0, resize image/mask pairs to a common resolution, and report road IoU on the official validation/test split. CamVid is a public driving-scene dataset, but it is not the Webots bird's-eye dataset used by Liu & Lu.
