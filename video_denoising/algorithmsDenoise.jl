include("prox_Dist.jl");
include("functions.jl")
include("startend.jl")

function deconvDistDenoise(im0, imbar, param, eta, alpha, alpha0, comm, sequence)

    nb_frame = 1;

    T = size(im0,1); ## T represente le nombre d'images total, normalement

    ## Synchronisation
    MPI.Barrier(comm);
    Nprocm = MPI.Comm_size(comm); ## <== Nombre total de procs
    labindex = MPI.Comm_rank(comm); ## <== Numero du proc en cours

    if labindex==0
        @printf("   Entering denoizing function !\n")
    end

    #load the motion matrices and their norms
    #@printf("   Starting bigStructMotionConstruction...\n")
    #@suppress begin
    if(alpha>0)
        structMotion = bigStructMotionConstruction(T,nb_frame,sequence);
    else
        structMotion = 0;
    end
    #end
    #@printf("   bigStructMotionConstruction complete !\n")

    yDual=[];

    ## CALCUL IDENTIQUE dans main : Peut etre passe en argument ?
    ## Repartition du nombre d'images par proc
    # partition the images


    im_start,im_end = startend_calc(labindex, Nprocm, T)


    ## Initialisation d'un array qui contient les images concernees
    ## Ce processeur en particulier
    im0_PerImage = im0[im_start: im_end];
    ## Nombre total d'images pour le processeur local 
    localT = im_end - im_start +1;

    if (labindex == 0 || labindex == Nprocm - 1) 
        @printf("   Process [%i] will be handling %i images \n", labindex, localT)
    end 

    
    #launch the program
    ## Lancement d'un timer, renvoye a la fin de la fonction
    #@printf("   Starting prox_Dist calculation...\n")
    time_end = @elapsed begin
        (xk, bigTime, SNR_mat, TOptim, TSynch) =  prox_Dist(im0_PerImage,
                                                            imbar,
                                                            param, 
                                                            nb_frame, 
                                                            eta, 
                                                            alpha, 
                                                            alpha0, 
                                                            structMotion, 
                                                            T, 
                                                            im_start, 
                                                            im_end, 
                                                            comm,
                                                            sequence)
    end
    @printf("prox_Dist calculation complete on [%i]!\n", labindex)


    # save the results in matlab format
    file = matopen("Out/TOptimLabindex=$(labindex)T=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "Optim$(Nprocm)_$(labindex)", TOptim)
    close(file)

    file = matopen("Out/TSynchLabindex=$(labindex)T=$(T)Nprocm=$(Nprocm).mat", "w")
    write(file, "Synch$(Nprocm)_$(labindex)", TSynch)
    close(file)

    xk = ivectorize(xk, localT, param["N1_x"], param["N2_x"]);
    im0_PerImage = copy(xk);


    bigRealTime = time_end;

    return xk, bigTime, bigRealTime, SNR_mat;
end

## TODO : Trouver signification de TSynch et TOptim dans la fonction suivante


