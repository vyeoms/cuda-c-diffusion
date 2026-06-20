#!/usr/bin/env python3
"""Download MNIST and save as raw float32 binary files for C consumption."""
import gzip
import os
import struct
import urllib.request
import numpy as np

URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "mnist_train_images.bin",
    "train-labels-idx1-ubyte.gz": "mnist_train_labels.bin",
    "t10k-images-idx3-ubyte.gz": "mnist_test_images.bin",
    "t10k-labels-idx1-ubyte.gz": "mnist_test_labels.bin",
}

def fetch_and_convert():
    os.makedirs("data", exist_ok=True)
    for gz_name, out_name in FILES.items():
        out_path = os.path.join("data", out_name)
        if os.path.exists(out_path):
            print(f"  {out_path} already exists, skipping")
            continue
        print(f"  downloading {gz_name} ...")
        urllib.request.urlretrieve(URL + gz_name, gz_name)
        with gzip.open(gz_name, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            n = struct.unpack(">I", f.read(4))[0]
            if magic == 2051:  # images
                rows = struct.unpack(">I", f.read(4))[0]
                cols = struct.unpack(">I", f.read(4))[0]
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
                data = data.astype(np.float32) / 255.0
            else:  # labels
                data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
            data.tofile(out_path)
            print(f"  wrote {out_path}  shape={data.shape}  dtype=float32")
        os.remove(gz_name)

if __name__ == "__main__":
    fetch_and_convert()
