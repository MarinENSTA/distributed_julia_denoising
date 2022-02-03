# Distributed Julia Denoising

## Authors

The base code was written by Feriel ABBOUD, all further modifications, updates and improvement were made by Marin STAMM

## About this project

This project aims to implement the distributed algorithm proposed here https://hal.archives-ouvertes.fr/hal-01942710v2, using the MPI library in Julia.

## Julia installation 
This is implemented in julia 1.4.1

Please install MPI and HDF5 on your machine before installing the following Julia packages 

To install necessary packages, open a julia terminal and type :
```julia
using Pkg         

Pkg.add("Images")
Pkg.add("HDF5")    
Pkg.add("MAT")    
Pkg.add("MPI")
Pkg.add("Suppressor")
```

To run this projet on your machine with four parallel processes using mpi: 
```bash
mpirun -np 4 julia programMain.jl
```


