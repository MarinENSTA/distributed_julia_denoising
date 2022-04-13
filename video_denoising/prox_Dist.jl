include("localOptimization.jl");
function prox_Dist(imblurnoisy, imbar, param, nb_frame, eta, alpha, alpha0, structMotion, T, im_start, im_end, comm, sequence)

    localT = size(imblurnoisy,1);

    ## Initialisation arbitraire.
    gammaj = 1.7;

    NbIt = param["NbIt"];
    N1_x = param["N1_x"];
    N2_x = param["N2_x"];
    xmin = 0;
    xmax = 1;

    bigTime = Array{Float32}(undef,NbIt);
    SNR_save = zeros(NbIt,localT);

    TOptim = Array{Float32}(undef,NbIt);
    TSynch = Array{Float32}(undef,NbIt);

    MPI.Barrier(comm);
    Nprocm = MPI.Comm_size(comm); ## <== Nombre total de processeurs.
    labindex = MPI.Comm_rank(comm); ## <== Numero du processeur local
    
    BigNormeAdj = Array{Array{Float32}}(undef, T);
    BigBetaArrayI = Array{Array{Float32}}(undef, T);


    #Handle the specific case in which the number of procs 
    #Does not divide the total number of images
    #offset = labindex*localT + 1 - im_start;


    if(alpha>0)
        for i in 1:T
            var = matread("$(sequence)/Motion/Beta$i.mat");
            BigBetaArrayI[i] = copy(var["betaArray"]);
        end
    end
    y =  Array{Array{Float32}}(undef,localT);
    # sauv is the number of previous images (0 or 1)
    # nb is the number of successive images (0 or 1)
    for k in 1:localT
        i = k + im_start - 1;
        if(alpha>0)
            if (i > nb_frame)
                nb = copy(nb_frame);
            else
                nb = i-1;
            end
            sauv = copy(nb);
            if (T- i >= nb_frame)
                nb = copy(nb_frame);
            else
                nb = T-i;
            end
        else
            sauv = 0; nb = 0;
        end
        y[k] = zeros((1+sauv+nb)*N1_x*N2_x);
    end

    x_tild =  vectorize(imblurnoisy, localT);

    xstruct = Array{Array{Float32}}(undef,localT);
    imq = zeros(N1_x*N2_x*2*localT);
    #println(size(imq));
    for k in 1:localT
        i = k + im_start - 1;
        if(alpha >0)
            if (i > nb_frame)
                nb = copy(nb_frame);
            else
                nb = i-1;
            end
            sauv = copy(nb);
            if (T- i >= nb_frame)
                nb = copy(nb_frame);
            else
                nb = T-i;
            end
        else
            sauv = 0; nb = 0;
        end

        #create our data, receive it for neighbor processors if needed
        tabSauv =  zeros(N1_x*N2_x*sauv);
        tabNb =  zeros(N1_x*N2_x*nb);
        for l in 1:sauv
            other_image = i-sauv+l-1;
            index_other_image = other_image - im_start +1;
            if(other_image >= im_start && other_image <= im_end)
                tabSauv[N1_x*N2_x*(l-1) + 1: N1_x*N2_x*l] = x_tild[N1_x*N2_x*(index_other_image-1)+1: N1_x*N2_x*index_other_image];
            else
                Nproc = labindex - Int64(ceil((im_start-other_image)*Nprocm/T));

                xxx = x_tild[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k];
                MPI.isend(xxx,Nproc,i,comm);
                xrec = zeros(N1_x*N2_x);
                (xrec, stat) = MPI.recv(Nproc,other_image,comm);
                tabSauv[N1_x*N2_x*(l-1) + 1: N1_x*N2_x*l] = xrec;
            end
        end

        for l in 1:nb
            other_image = i+nb-l+1;
            index_other_image = other_image - im_start +1;
            if(other_image >= im_start && other_image <= im_end)
                tabNb[N1_x*N2_x*(l-1) + 1: N1_x*N2_x*l] = x_tild[N1_x*N2_x*(index_other_image-1)+1: N1_x*N2_x*index_other_image];
            else
                Nproc = labindex + Int64(ceil((other_image-im_end)*Nprocm/T));

                MPI.isend(x_tild[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k],Nproc,i,comm);
                xrec = zeros(N1_x*N2_x);
                (xrec, stat) = MPI.recv(Nproc,other_image,comm);
                tabNb[N1_x*N2_x*(l-1) + 1: N1_x*N2_x*l] = xrec;
 
            end
        end

        xstruct[k] = [x_tild[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k]; tabSauv; tabNb] - y[k];
        imq[N1_x*N2_x*2*(k-1)+1:N1_x*N2_x*2*k] = ComputeTVlin_alpha(x_tild[N1_x*N2_x*(k-1)+1:N1_x*N2_x*k], N1_x, N2_x);
    end

    x = zeros(N1_x*N2_x*localT);
    freqSynch = 4; #4
    #Launch the algorithm
    for indB = 1: NbIt
        time1_end = @elapsed begin
            #Local optimization
            #@suppress begin
                timeO_end = @elapsed begin
                    localOptimization(xstruct, imq, y, eta, alpha, alpha0, structMotion, BigBetaArrayI, N1_x, N2_x, xmin,xmax, gammaj, T, nb_frame, im_start, im_end, sequence);
                end
            #end
            TOptim[indB] = timeO_end

            timeS_end = @elapsed begin
                x = zeros(N1_x*N2_x*localT);
                bigTabSauv = zeros(N1_x*N2_x);
                bigTabNb = zeros(N1_x*N2_x);
                #------Summation
                for i in im_start: im_end
                    if(alpha>0)
                        if (i> nb_frame)
                            nb = copy(nb_frame);
                        else
                            nb = i-1;
                        end
                        sauv = copy(nb);
                        if (T- i >= nb_frame)
                            nb = copy(nb_frame);
                        else
                            nb = T-i;
                        end
                    else
                        sauv = 0; nb = 0;
                    end
                    k = i -im_start + 1;
                    x[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k] += xstruct[k][1:N1_x*N2_x];
                    for l in 1:sauv
                        other_image = i-sauv+l-1;
                        if(other_image >= im_start && other_image <= im_end)
                            shift_other_image = other_image - im_start +1;
                            if (other_image > nb_frame)
                                sauv_other = copy(nb_frame);
                            else
                                sauv_other = other_image-1;
                            end
                            x[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k] += xstruct[shift_other_image][N1_x*N2_x*(l+sauv_other) + 1: N1_x*N2_x*(l+sauv_other+ 1)];
                        end
                    end

                    for l in 1:nb
                        other_image = i+nb-l+1;
                        if(other_image >= im_start && other_image <= im_end)
                            shift_other_image = other_image - im_start +1;
                            x[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k] += xstruct[shift_other_image][N1_x*N2_x*l + 1: N1_x*N2_x*(l+ 1)];
                        end
                    end
                end

                if(indB%freqSynch!=0) #----Local optimization ( no need for send/receive process)
                    if labindex==0
                        @printf("local proc [%i] entering local optimisation step\n",labindex);
                    end
                    if(alpha>0)
                        for i in im_start: im_end
                            k = i -im_start + 1;

                            if (i> nb_frame)
                                nb = copy(nb_frame);
                            else
                                nb = i-1;
                            end
                            sauv = copy(nb);
                            if (T- i >= nb_frame)
                                nb = copy(nb_frame);
                            else
                                nb = T-i;
                            end

                            if((k==1 || k==localT)) # this test is necessary because k=1 doesn't mean i = 1 and thus sauv !=0
                                x[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k]/= 2;
                            else
                                x[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k]/= (1+sauv+nb);
                            end
                        end
                        # else nonthing /1
                    end

                else #----Global optimization
                    if labindex == 0
                        @printf("local proc [%i] entering global optimisation step\n",labindex);
                    end 

                    if(alpha>0)
                        if(labindex>0)
                            bigTabSauv = xstruct[1][N1_x*N2_x+1:2*N1_x*N2_x];
                        end
                        if(labindex<Nprocm-1)
                            bigTabNb= xstruct[localT][2*N1_x*N2_x+1:end];
                        end
                    end

                    if(labindex<Nprocm-1) # send summantion and bigTabNb to c+1
                        xtemp = [x[N1_x*N2_x*(localT-1)+1:end]; bigTabNb];
                        MPI.Send(xtemp,labindex+1, 100+labindex,comm);
                    end
                    if(labindex>0) # receive summantion and data to add to bigTabSauv from c-1
                        xrec = zeros(2*N1_x*N2_x);
                        MPI.Recv!(xrec,labindex-1,100+labindex-1,comm);
                        bigTabSauv += xrec[1:N1_x*N2_x];
                        x[1:N1_x*N2_x]+= xrec[N1_x*N2_x+1:N1_x*N2_x*2];
                    end
                    for i in im_start: im_end
                        k = i -im_start + 1;
                        if(alpha>0)
                            if (i> nb_frame)
                                nb = copy(nb_frame);
                            else
                                nb = i-1;
                            end
                            sauv = copy(nb);
                            if (T- i >= nb_frame)
                                nb = copy(nb_frame);
                            else
                                nb = T-i;
                            end
                        else
                            sauv = 0; nb = 0;
                        end
                        x[N1_x*N2_x*(k-1)+1: N1_x*N2_x*k]/= (1+sauv+nb);
                    end
                    if(alpha>0)
                        if(im_start==2) #TabSauvi corresponds to the first image (i=1)
                            bigTabSauv/=2;
                        else
                            bigTabSauv/=3;
                        end
                    end

                    if(labindex>0) #send averaged TabSauv and the first image to c-1
                        xtemp = [bigTabSauv; x[1:N1_x*N2_x]];
                        MPI.Send(xtemp,labindex-1,300+labindex,comm);
                    end

                    if(labindex<Nprocm-1) #receive the last image and TabNb from c+1
                        xrec = zeros(2*N1_x*N2_x);
                        MPI.Recv!(xrec,labindex+1,300+labindex+1,comm);
                        x[N1_x*N2_x*(localT-1)+1:end] = xrec[1:N1_x*N2_x];
                        bigTabNb = xrec[N1_x*N2_x+1:N1_x*N2_x*2];
                    end
                end
                #-------Update
                for i in im_start: im_end
                    if(alpha >0)
                        if (i > nb_frame)
                            nb = copy(nb_frame);
                        else
                            nb = i-1;
                        end
                        sauv = copy(nb);
                        if (T- i >= nb_frame)
                            nb = copy(nb_frame);
                        else
                            nb = T-i;
                        end
                        nu = 1/3;
                    else
                        sauv = 0; nb = 0; nu =1;
                    end
                    k = i -im_start +1;
                    wj = 1/(1+sauv+nb);
                    xstruct[k][1:N1_x*N2_x] += gammaj*nu*wj^(-1)*(x[N1_x*N2_x*(k-1)+1:N1_x*N2_x*k] -  xstruct[k][1:N1_x*N2_x]);

                    for l in 1:sauv
                        other_image = i-sauv+l-1;
                        if(other_image==1)
                            wj = 1/2;
                        else
                            wj = 1/3;
                        end
                        if(other_image >= im_start && other_image <= im_end)
                            shift_other_image = other_image - im_start +1;
                            xstruct[k][N1_x*N2_x*l+1: N1_x*N2_x*(l+1)] += gammaj*nu*wj^(-1)*(x[N1_x*N2_x*(shift_other_image-1)+1: N1_x*N2_x*shift_other_image] -  xstruct[k][N1_x*N2_x*l+1: N1_x*N2_x*(l+1)]);
                        else
                            if(indB % freqSynch==0) #global synch
                                xstruct[k][N1_x*N2_x*l+1: N1_x*N2_x*(l+1)] += gammaj*nu*wj^(-1)*(bigTabSauv -xstruct[k][N1_x*N2_x*l+1: N1_x*N2_x*(l+1)]);
                            end
                        end
                    end
                    for l in 1:nb
                        other_image = i+nb-l+1;
                        if(other_image==T)
                            wj = 1/2;
                        else
                            wj = 1/3;
                        end
                        if(other_image >= im_start && other_image <= im_end)
                            shift_other_image = other_image - im_start +1;
                            xstruct[k][N1_x*N2_x*(l+sauv)+1: N1_x*N2_x*(l+sauv+1)]  += gammaj*nu*wj^(-1)*(x[N1_x*N2_x*(shift_other_image-1)+1: N1_x*N2_x*shift_other_image] -  xstruct[k][N1_x*N2_x*(l+sauv)+1: N1_x*N2_x*(l+sauv+1)]);
                        else
                            if(indB % freqSynch==0) #global synch
                                xstruct[k][N1_x*N2_x*(l+sauv)+1: N1_x*N2_x*(l+sauv+1)]  += gammaj*nu*wj^(-1)*(bigTabNb - xstruct[k][N1_x*N2_x*(l+sauv)+1: N1_x*N2_x*(l+sauv+1)]);
                            end
                        end
                    end
                end
            end # Regarder timeS_end (pics observés ??)
            TSynch[indB] = timeS_end;
        end
        bigTime[indB] =time1_end;

        for ii = 1: localT ## Mettre 0 a la place et observer ce qui se passe
            SNR_save[indB,ii] = 10*log10(sum(imbar[im_start - 1 + ii][:].^(2)) / sum( (imbar[im_start - 1 + ii][:] - x[N1_x*N2_x*(ii-1)+1: N1_x*N2_x*ii]).^(2)));
        end

        # if(labindex == 6)
        #     println("Step done on proc [4]")
        #     println("it = $(indB)");
        #     @printf("SNR = %f \n",SNR_save[indB,1]);
        #     @printf("time = %f \n",time1_end);
        # end
        if(labindex == 0)
            println("Step done on proc [0]")
            println("it = $(indB)");
            @printf("SNR = %f \n",SNR_save[indB,1]);
            @printf("time = %f \n",time1_end);
            #AJOUTER un print timeS
        end
    end

    return x, bigTime, SNR_save, TOptim, TSynch;

end
