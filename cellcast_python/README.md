# cellcast_python

Cellcast Python bindings for the Rust core library.

## Build cellcast_python from source

To build the cellcat_python package from source, use the `maturin` build tool
(this requires the Rust toolchain). If you're using `uv` to manage your Python
virtual environments (venv) add `numpy` and `maturin` to your environment and run the
`maturin develop --release` command in the `cellcast_python` directory of the
[cellcast](https://github.com/uw-loci/cellcast) repository with your venv activated:

```bash
$ source ~/path/to/myenv/.venv/bin/activate
$ (myenv) cd cellcast_python
$ maturin develop --release
```

Alernatively if you're using `conda` or `mamba` you can do the following:

```bash
$ cd cellcat_python
$ mamba activate myenv
(myenv) $ mamba install numpy maturin
...
(myenv) $ maturin develop --release
```

This will install cellcast in the currently active Python environment.

### Using `cellcast`

Once cellcast_python  has been installed in a compatiable Python environment, `cellcast`
will be available to import. The following example demonstrates how to perform
model inference with cellcast:

```python
import cellcast.models as ccm
from tifffile import imread

# load 2D data for inference
data = imread("path/to/data.tif")

# run stardist inference and produce instance segmentations
labels = ccm.stardist_2d_versatile_fluo.predict(data)
```
