# Compare PyTorch affine transform with OpenCV's one

## Set environment  

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Compare  

```
python verify.py
```

This command outputs `output_numpy.png`, `output_torch.png` and `diff.png` in `data` directory.  

## Reference  

* https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/19
