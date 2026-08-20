#!/usr/bin/env python3

import h5py
import numpy as np
import sys

def inspect_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- Inspeccionando archivo: {file_path} ---")
            for key in f.keys():
                data = f[key]
                print(f"Dataset: {key}")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                if len(data) > 0:
                    print(f"  Ejemplo (primeras 2 filas):\n{data[:2]}")
                print("-" * 30)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 inspect_h5.py <archivo.h5>")
    else:
        inspect_h5(sys.argv[1])
