
import Images, MAT, HDF5, Suppressor, Printf, SparseArrays
using Images, MAT, HDF5
using MPI
using Suppressor
using Printf
using SparseArrays

include("functions.jl")
include("algorithmsDenoise.jl")
include("startend.jl")

#### SEQUENCE ####
sequence = "Flower";
# PARAMETERS (PLEASE REFER TO PARAMETERS.TXT FILE FOR EXEMPLE MEDIA)
eta = 0.029514;
alpha = 0.060166;
alpha0 = 0.91304;

#### images #####
T = 72;


img  = Array{Array{Float32}}(undef, T);
im  = Array{Array{Float32}}(undef, T);
imblurnoisy = Array{Array{Float32}}(undef, T);



for i in 1:T
    II = matread("../media/$(sequence)/ForJuliaDenoise/$(lowercase(sequence))HR$i.mat");
    I = copy(II["I"]);
    im[i] = I;

    II = matread("../media/$(sequence)/ForJuliaDenoise/$(lowercase(sequence))LR$i.mat");
    I = copy(II["IBlur"]);
    imblurnoisy[i] = I;
    img[i] = I;
end


# Image dimensions 
(N1_x, N2_x) = size(im[1]);

# Initialising Param dictionnary... 
param = Dict(); 
param["NbIt"] = 30; 
param["N1_x"] = N1_x;
param["N2_x"] = N2_x;

# Initialising mPI
MPI.Init();
comm = MPI.COMM_WORLD;
labindex = MPI.Comm_rank(comm); ## <=== LABINDEX = Proc rank
Nprocm = MPI.Comm_size(comm); ## <==== Total proc number

if labindex==0
    @printf("There are %i parallel processes initialized\n",Nprocm)
    @printf("----------   Starting ---------- \n");
end


(imnew, time, realTime, SNR_mat) = deconvDistDenoise(img,im,param,eta,alpha,alpha0,comm,sequence);

xi = vectorize(imnew, size(imnew,1));

x = zeros(N1_x*N2_x*T);
globalRealTime = MPI.Reduce(realTime, MPI.MAX, 0, comm);

MPI.Barrier(comm);
globalTime = MPI.Reduce(time, MPI.MAX, 0, comm);

# save results in julia format
## SNR = signal to noise ratio
h5open("Out/mySNRPerImageLabindex=$(labindex)T=$(T)Nprocm=$(Nprocm).h5","w") do file
    write(file,"SNR_mat", SNR_mat)
end

# save results in matlab format
file = matopen("Out/mySNRPerImageLabindex=$(labindex)T=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "SNR_mat$(Nprocm)_$(labindex)", SNR_mat)
    close(file)

##Synchronisation
MPI.Barrier(comm);

# On proc 0
if(labindex==0)

    im_start,im_end = startend_calc(labindex, Nprocm, T)


    x[N1_x*N2_x*(im_start -1)+1: N1_x*N2_x*im_end] += xi;

    for Nproc = 1:Nprocm-1
        im_startProc,im_endProc = startend_calc(Nproc, Nprocm, T)

        
        xii = zeros(N1_x*N2_x*(im_endProc - im_startProc +1));
        (xii, stat) = MPI.recv(Nproc, Nproc, comm);
        x[N1_x*N2_x*(im_startProc -1)+1: N1_x*N2_x*im_endProc] += xii;
    end

    SNR = 10*log10(sum( vectorize(im, T).^(2) ) / sum( (vectorize(im, T) - x).^(2)));
    @printf("\n \n");
    @printf("\nSNR final= %f \n",SNR);
    @printf("Temps global = %f \n",sum(globalTime));
    @printf("Temps reel global = %f \n",sum(globalRealTime));
    
    h5open("Out/myTime=$(T)Nprocm=$(Nprocm).h5","w") do file
        write(file,"time", globalTime)
    end
    file = matopen("Out/myTime=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "time", globalTime)
    close(file)

    h5open("Out/myRealTime=$(T)Nprocm=$(Nprocm).h5","w") do file
        write(file,"realTime", globalRealTime)
    end
    file = matopen("Out/myRealTime=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "realTime",globalRealTime)
    close(file)

    h5open("Out/mySNR=$(T)Nprocm=$(Nprocm).h5","w") do file
        write(file,"SNR",SNR)
    end
    file = matopen("Out/Ireneimnew$(Nprocm).mat", "w")
    write(file, "imnew", x)
    close(file)

else
    MPI.isend(xi,0,labindex,comm);
end

MPI.Barrier(comm);
MPI.Finalize();


