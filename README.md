# HydrogelMPM
![Github License](https://img.shields.io/github/license/Yihao-Shi/Hydrogel)

A Material point method for simulating hydeogel materials, which is developed by [Taichi language](https://github.com/taichi-dev/taichi).

## Demo
<p align="center">
    <img src="https://github.com/Yihao-Shi/GeoTaichi/blob/main/hydrogel.gif" width="50%" height="50%" />
</p>

## How to install
Our code is compatible with Python3.8 ~ Python3.11.
```
python -m pip install taichi==1.6 numpy pyevtk
```

## Run examples
```
python hydrogelImpact.py
```

## Features
1. Updated/Total Largrangian scheme
2. Talyor PIC transfer scheme
3. PIC/FLIP mapping
4. F bar method for nearly incompressible materials
5. Five hyperelastic models:
   - Neo-hookean
   - Mooney Rivlin
   - Generalized Mooney Rivlin
   - Gent
   - Gent-Gent
7. Simple GUI
