
import Images, MAT, HDF5, Suppressor, Printf, SparseArrays
using Images, MAT, HDF5
using MPI
using Suppressor
using Printf
using SparseArrays

#@suppress begin #to suppress warnings related to julia >= version 0.5.0
include("functions.jl")
include("algorithmsDenoise.jl")
include("startend.jl")
#end

#### images #####
T = 72;

# A = Array(Array{Float32}, 10)
##Initialisation des tableaux des images
img  = Array{Array{Float32}}(undef, T);
im  = Array{Array{Float32}}(undef, T);
imblurnoisy = Array{Array{Float32}}(undef, T);



#Foreman
#for i in 1:T
#     II = matread("Foreman/ForJuliaDenoise/ForemanHR$i.mat");
#     I = copy(II["I"]);
#     im[i] = I;

#     II = matread("Foreman/ForJuliaDenoise/ForemanLR$i.mat");
#     I = copy(II["IBlur"]);
#     imblurnoisy[i]=I;
#     img[i] = I;
#end

# Claire
for i in 1:T
    II = matread("Claire/ForJuliaDenoise/ClaireHR$i.mat");
    I = copy(II["I"]);
    im[i] = I;

    II = matread("Claire/ForJuliaDenoise/ClaireLR$i.mat");
    I = copy(II["IBlur"]);
    imblurnoisy[i] = I;
    img[i] = I;
end

##T : Nombre d'images
#@printf ("Loading images...")
#for i in 1:T
#     #Telechargement image non bruitee
#     #@printf(" i  : %i\n", i);
#     II = matread("Irene/ForJuliaDenoise/IreneHR$i.mat");
#     I = copy(II["I"]);
#     #push!(im,I);
#     im[i] = I;

     ##Telechargement image bruitee
#     II = matread("Irene/ForJuliaDenoise/IreneLR$i.mat");
#     I = copy(II["IBlur"]);
#     #push!(imblurnoisy,I);
#     #push!(img,I);

#     imblurnoisy[i] = I;
#     img[i] = I;

#end


#@printf (" ==> loading complete ! \n")


## Recupation des dimensions de l'image ?
(N1_x, N2_x) = size(im[1]);

#println(N1_x)
#println(N2_x)
#@printf ("Initialising Param dictionnary... ")
param = Dict(); ## Initialisation d'un Dictionnaire ! 
## Ce dictionnaire a pour unique but d'etre envoye dans 'deconvDistDenoise'
param["NbIt"] = 30; #60
param["N1_x"] = N1_x;
param["N2_x"] = N2_x;
#@printf (" ==> Initialisation complete ! \n")

#Foreman
#eta = 8e-2;
#alpha = 2e-2;
#alpha0 = 0.7108;

# #Claire
eta = 3.2e-2;
alpha = 4.5e-2;
alpha0 = 0.7108;

## Parametres propres aux methodes appliquees sur Irene
#Irene
## Table 1 page (paramètre beta dans l'article) (Page 33)
## eta : régularisation spatiale
## Alpha 0 : comment on pénalise les termes de mouvement de manière extrème 
## Alpha 
#eta = 2.0444e-2;
#alpha = 1.5e-2;
#alpha0 = 0.7108;

#@printf ("Initialising MPI... ")
##Initialisation MPI
MPI.Init();
comm = MPI.COMM_WORLD;
labindex = MPI.Comm_rank(comm); ## <=== LABINDEX = RANK PROC
Nprocm = MPI.Comm_size(comm); ## <==== Nombre total de processeurs
#@printf (" ==> Initialisation complete : \n")
#println(Nprocm, labindex)
if labindex==0
    @printf("There are %i parallel processes initialized\n",Nprocm)
end


#@printf('[%i] : lancement du calcul', labindex);

if labindex==0
    @printf("----------   Starting ---------- \n");
end

#@suppress begin
## Calcul de denoizing ?
#@printf ("Starting denoizing calculation...\n")
(imnew, time, realTime, SNR_mat) = deconvDistDenoise(img,im,param,eta,alpha,alpha0,comm);
#@printf (" ==> Calculation complete ! \n")

# if labindex==0
#     println("img : ",typeof(img));
#     println("im  : ",typeof(im));
#     println("param : ",typeof(param));
#     println("imnew : ",typeof(imnew));
#     println("time : ",typeof(time));
#     println("realTime : ",typeof(realTime));
#     println("SNR_Mat : ",typeof(SNR_mat));

# end
xi = vectorize(imnew, size(imnew,1));

#end

## Initialisation d'une matrice de zeros a la bonne taille
x = zeros(N1_x*N2_x*T);
##Recuperation des temps, et recherche du realtemps maximum
globalRealTime = MPI.Reduce(realTime, MPI.MAX, 0, comm);
##Etape de synchronisation... Indispensable ? Reduce n'est pas deja synchronisant ?
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

## Si on est sur le proc 0
if(labindex==0)

    im_start,im_end = startend_calc(labindex, Nprocm, T)


    x[N1_x*N2_x*(im_start -1)+1: N1_x*N2_x*im_end] += xi;

    for Nproc = 1:Nprocm-1
        im_startProc,im_endProc = startend_calc(Nproc, Nprocm, T)

        
        ## Initialisation d'une matrice a la bonne taille
        xii = zeros(N1_x*N2_x*(im_endProc - im_startProc +1));
        ## Insertion du resultat dans cette matrice a la bonne taille
        ## Reception de tous les processeurs de 1 a nb_proc - 1
        (xii, stat) = MPI.recv(Nproc, Nproc, comm);
        ## Insertion du resultat recu au bon endroit dans la matrice globale
        x[N1_x*N2_x*(im_startProc -1)+1: N1_x*N2_x*im_endProc] += xii;
    end

    #Affichages globaux post calculs
    ## Calcul du SNR
    SNR = 10*log10(sum( vectorize(im, T).^(2) ) / sum( (vectorize(im, T) - x).^(2)));
    @printf("\n \n");
    @printf("\nSNR final= %f \n",SNR);
    @printf("Temps global = %f \n",sum(globalTime));
    @printf("Temps reel global = %f \n",sum(globalRealTime));
    
    ##Ecriture dans un fichier julia du temps global
    h5open("Out/myTime=$(T)Nprocm=$(Nprocm).h5","w") do file
        write(file,"time", globalTime)
    end

    ##Ecriture dans un fichier matlab du temps global.
    file = matopen("Out/myTime=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "time", globalTime)
    close(file)

    ##Ecriture dans un fichier julia du realtemps global.
    h5open("Out/myRealTime=$(T)Nprocm=$(Nprocm).h5","w") do file
        write(file,"realTime", globalRealTime)
    end

    ## Ecriture dans un fichier matlab du realtemps global.
    file = matopen("Out/myRealTime=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "realTime",globalRealTime)
    close(file)

    ## Ecriture dans un fichier Julia du SNR
    h5open("Out/mySNR=$(T)Nprocm=$(Nprocm).h5","w") do file
        write(file,"SNR",SNR)
    end

    ## Ecriture dans un fichier Matlab du SNR
    file = matopen("Out/Ireneimnew$(Nprocm).mat", "w")
    write(file, "imnew", x)
    close(file)

## SI l'ON N'EST PAS LE PROC 0, ON ENVOIE SON RESULTAT
## De maniere non bloquante....
else
    MPI.isend(xi,0,labindex,comm);
end

## .... Mais on bloque quand meme juste apres ! 
## Pourrait-t-on mettre un envoi bloquant tout simplement ? 
## On est a la fin du code, plus d'asynchronisme necessaire, semble-t-il

MPI.Barrier(comm);
MPI.Finalize()

############# Pour les read ###################
# Exemple
# time24 = h5open("myRealTime=72Nprocm=24.h5","r") do file read(file, "realTime") end




