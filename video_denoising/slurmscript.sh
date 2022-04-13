#!/bin/bash


#SBATCH --job-name=bus_smooth
#SBATCH --output=flower_Smooth_Denoising_30_30.out
#SBATCH --time=1:00:00
#SBATCH --ntasks=36
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH --partition=cpu_short

module purge 
module load julia/1.4.0/gcc-9.2.0
module load hdf5/1.10.6/intel-19.0.3.199-intel-mpi
module load intel-mpi/2019.3.199/intel-19.0.3.199
export JULIA_MPI_CLUSTER_WARN=n


cd /gpfs/users/stammma/distributed_julia_denoising/video_denoising

#Job
set -x

END=36;
for numprocs in $(seq 36 $END)
do
	start=$SECONDS
	echo "Simulation en cours pour NumProcs = $numprocs ..."
	mpirun -np $numprocs julia programMain.jl
	echo "numprocs = $numprocs done
	"

	duration=$(( SECONDS - start ))
	echo " Total completion time : $duration"
done
