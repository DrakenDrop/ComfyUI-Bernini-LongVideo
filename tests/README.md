# CPU tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

Requires NumPy and Pillow. Tests use synthetic frames and mocked HTTP responses; they do not validate real GPU inference or visual seam quality.
