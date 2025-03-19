#!/bin/bash

# This script is used to build the toolkit

# Check if g++ is installed
if ! command -v g++ &> /dev/null
then
    echo "g++ could not be found"
    exit 1
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found"
    exit 1
fi

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null
then
    echo "CUDA could not be found"
    exit 1
fi

# Check if Metal is installed (macOS specific)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v xcrun &> /dev/null || ! xcrun -sdk macosx metal &> /dev/null
    then
        echo "Metal could not be found"
        exit 1
    fi
fi

echo "All required tools are installed."

# Build the toolkit

mkdir -p build/toolkit/{lib,include,bin}
python3 setup.py build_ext --inplace
g++ -shared -o LibMoudoPyCxx99.so MouseLib.c -fPIC -I/usr/include/python3.8
pyinstaller --onefile --clean --noconfirm --distpath ./build/toolkit/bin --name moucmodder miceModder.py
pyinstaller --onefile --clean --noconfirm --distpath ./build/toolkit/bin --name MoudoConfigurator src/cof/main.py
cp LibMoudoPyCxx99.so ./build/toolkit/lib
cp MouseLib.h ./build/toolkit/include
nsis toolkit.nsi