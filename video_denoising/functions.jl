function kernel_sinc()
    x= (-6:22)';
    h= sinc(x./pi);
    h= hcat(zeros(1,20), h)
    h = hcat(h, zeros(1,4));
    h= h./sum(h);
    return h;
end

function oddDecimation(x)
    p = copy(x[1:2:end-1,:,:]);
    return p;
end

function oddDecimationAdj(x)
    (N1_x,N2_x)= size(x,1,2);
    N3_x = size(x,3);
    p = zeros(2*N1_x,N2_x,N3_x);
    p[1:2:end-1,:,:] = x;
    return p;
end

function evenDecimation(x)
    p = copy(x[2:2:end,:,:]);
    return p;
end

function evenDecimationAdj(x)
    (N1_x,N2_x)= size(x,1,2);
    N3_x = size(x,3);
    p = zeros(2*N1_x,N2_x,N3_x);
    p[2:2:end,:,:] = x;
    return p;
end

function decimation2D(x)
    p = copy(x[1:2:end,1:2:end,:]);
    return p;
end

function decimation2DAdj(x)
    (N1_x,N2_x)= size(x,1,2);
    N3_x = size(x,3);
    p = zeros(2*N1_x,2*N2_x,N3_x);
    p[1:2:end,1:2:end,:] = x;
    return p;
end

function oddinterpolation(x)
    xf = filt(ones(2),1,x); #sommer chaque 2 lignes
    xff = filt(ones(3),1,xf')'./6; #sommer chaque 3 colonnes /6
    xff[:,1:end-1] = xff[:,2:end];
    (Nx, Ny) = size(x);
    #supprimer la premi?re ligne
    supp = copy(xff[end,:]);
    y = zeros(2*Nx, Ny);
    y[1:2:end, :]= copy(x);
    y[2:2:end-2, :] = xff[2:end,:];
    y[end,:] = supp;
    return y;
end

function eveninterpolation(x)
    xf = filt(ones(2),1,x); #sommer chaque 2 lignes
    xff = filt(ones(3),1,xf')'./6; # sommer chaque 3 colonnes /6
    xff[:,1:end-1] = xff[:,2:end];
    (Nx, Ny) = size(x);
    #supprimer la premi?re ligne
    supp = copy(xff[2,:]);
    y = zeros(2*Nx, Ny);
    y[2:2:end, :]= copy(x);
    y[3:2:end, :] = xff[2:end,:];
    y[1,:] = supp;
    return y;
end

function sauvNbComput(i, T, nb_frame)
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
    return sauv,nb;
end

## VUE DANS algorithmDenoise.jl
## Appel :  structMotion = bigStructMotionConstruction(T, nb_frame);
## T = nombre total d'images
## nb_frame = 1 ("Numero de l'image de depart ?")
function  bigStructMotionConstruction(sizeIm, nb_frame,sequence)

    labindex = MPI.Comm_rank(comm);
    if labindex==0
        @printf("       Entering bigStructMotionConstruction !")
    end
    structMotion = Array{Any}(undef, sizeIm);
    indexTab= Array{Float32}(undef, 2*nb_frame);
    motionTab = Array{SparseMatrixCSC{Float32,Int64}}(undef, 2*nb_frame);
    normeTab= Array{Float32}(undef,2*nb_frame);

    ##Dictionnaire
    dict = Dict();

    for i in 1: sizeIm
        if (i > nb_frame)
            nb = copy(nb_frame);
        else
            nb = i-1;
        end
        sauv = copy(nb);
        if (sizeIm- i >= nb_frame)
            nb = copy(nb_frame);
        else
            nb = sizeIm-i;
        end

        for j in 1:sauv
            indexTab[j] = (i-sauv+j-1);

            var = matread("$(sequence)/Motion/M$(i-sauv+j-1)$(i).mat");
            motionTab[j] = copy(var["Pmat"]);

            var = matread("$(sequence)/Motion/Norm$(i-sauv+j-1)$(i).mat");
            normeTab[j] = copy(var["norme"]);

        end

        for j = 1:nb
            indexTab[j+sauv] = (i+nb-j+1);

            var = matread("$(sequence)/Motion/M$(i+nb-j+1)$(i).mat");
            motionTab[j+sauv] = copy(var["Pmat"]);

            var = matread("$(sequence)/Motion/Norm$(i+nb-j+1)$(i).mat");
            normeTab[j+sauv] = copy(var["norme"]);

        end

        dict["Index"] = copy(indexTab);
        dict["Motion"] = copy(motionTab);
        dict["Norme"] = copy(normeTab);

        ##Chaque element de structMotion est un dictionnaire 
        ## Contenant Index, Motion, et norme.
        structMotion[i] = copy(dict);

    end

    if labindex==0
        @printf(" ==> Done ! \n")
    end
    
    return structMotion;
end



function motionAdjCompPerImage(sizeIm, nb_frame, i, structMotion)
    #[normeAdj, MatAdj]
    indexTab= Array{Float32}(undef,2*nb_frame);
    motionTab = Array{SparseMatrixCSC{Float32,Int32}}(undef, 2*nb_frame);
    normeTab= Array{Float32}(undef,2*nb_frame);

    if (i > nb_frame)
        nb = copy(nb_frame);
    else
        nb = i-1;
    end
    sauv = copy(nb);
    if (sizeIm - i >= nb_frame)
        nb = copy(nb_frame);
    else
        nb = sizeIm - i;
    end

    MatAdj = Array{SparseMatrixCSC{Float32,Int32}}(undef, sauv+nb);
    normeAdj= Array{Float32}(undef,sauv+nb);

    for j = 1: sauv
        indexTab = copy(structMotion[i-sauv+j-1]["Index"]);
        motionTab = copy(structMotion[i-sauv+j-1]["Motion"]);
        normeTab = copy(structMotion[i-sauv+j-1]["Norme"]);

        ind = find( x->(x == i), indexTab)[1];

        normeAdj[j] = copy(normeTab[ind]);
        MatAdj[j] = copy(motionTab[ind]);
    end


    for j = 1:nb
        indexTab = copy(structMotion[i+nb-j+1]["Index"]);
        motionTab = copy(structMotion[i+nb-j+1]["Motion"]);
        normeTab = copy(structMotion[i+nb-j+1]["Norme"]);

        ind = find( x->(x == i), indexTab)[1];

        normeAdj[j+sauv] = copy(normeTab[ind]);
        MatAdj[j+sauv] = copy(motionTab[ind]);

    end

    return normeAdj, MatAdj;
end

function gradh1(x, nx, ny)
    X = reshape(x,nx,ny);
    D = copy(X);
    D[:,2:ny] = X[:,2:ny]-X[:,1:(ny-1)];
    return D;
end

function gradv1(x, nx, ny)
    X = reshape(x,nx,ny);
    D = copy(X);
    D[2:nx,:] = X[2:nx,:]-X[1:(nx-1),:];
    return D;
end

function gradh2(x, nx, ny)

    X = reshape(x,nx,ny);
    D = copy(X);
    D[:,3:ny] = X[:,3:ny]-X[:,1:(ny-2)];
    return D;
end

function gradv2(x, nx, ny)
    X = reshape(x,nx,ny);
    D = copy(X);
    D[3:nx,:] = X[3:nx,:]-X[1:(nx-2),:];
    return D;
end

function gradh1_adj(z,nx,ny)
    Dt = copy(z);
    Dt[:,1:(ny-1)] = z[:,1:(ny-1)]-z[:,2:ny];
    return Dt;
end

function gradv1_adj(z,nx,ny)
    Dt = copy(z);
    Dt[1:(nx-1),:] = z[1:(nx-1),:]-z[2:nx,:];
    return Dt;
end

function gradh2_adj(z,nx,ny)
    Dt = copy(z);
    Dt[:,1:(ny-2)] = z[:,1:(ny-2)]-z[:,3:ny];
    return Dt;
end

function gradv2_adj(z,nx,ny)
    Dt = copy(z);
    Dt[1:(nx-2),:] = z[1:(nx-2),:]-z[3:nx,:];
    return Dt;
end

function gradh1v1haut(x, nx, ny)
    X = reshape(x,nx,ny);
    D = copy(X);
    D[2:nx,2:ny] = X[2:nx,2:ny]-X[1:(nx-1),1:(ny-1)]; #horizontal
    return D;
end

function gradh1v1bas(x, nx, ny)
    X = reshape(x,nx,ny);
    D = copy(X);
    D[1:(nx-1),2:ny] = X[1:(nx-1),2:ny]-X[2:nx,1:(ny-1)]; #horizontal
    return D;
end

function gradh1v1haut_adj(z,nx,ny)
    Dt = copy(z);
    Dt[1:(nx-1),1:(ny-1)] = z[1:(nx-1),1:(ny-1)]-z[2:nx,2:ny];
    return Dt;
end

function gradh1v1bas_adj(z,nx,ny)
    Dt = copy(z);
    Dt[2:nx,1:(ny-1)] = z[2:nx,1:(ny-1)]-z[1:(nx-1),2:ny];
    return Dt;
end

function ComputeSltv1(x,nx,ny)
    norme = 5.65;

    dh1(x) = gradh1(x,nx,ny);
    dv1(x) = gradv1(x,nx,ny);

    Du = dh1(dh1(x));
    Dv = dh1(dv1(x));

    Dx = vcat(Du[:], Dv[:])./norme;
    return Dx;
end

function ComputeSltv2(x,nx,ny)
    norme = 4.95;

    dh1(x) = gradh1(x,nx,ny);
    dv1(x) = gradv1(x,nx,ny);
    dh2(x) = gradh2(x,nx,ny);

    Du = dh2(dh1(x));
    Dv = dh2(dv1(x));

    Dx = vcat(Du[:], Dv[:])./norme;
    return Dx;
end

function ComputeSltv4(x,nx,ny)

    # 1er voisin vertical haut
    norme = 5.65;

    dh1(x) = gradh1(x,nx,ny);
    dv1(x) = gradv1(x,nx,ny);

    Du = dv1(dh1(x));
    Dv = dv1(dv1(x));

    Dx = vcat(Du[:], Dv[:])./norme;
    return Dx;
end

function ComputeSltv5(x,nx,ny)

    # 2eme voisin vertical haut
    norme = 4.95;

    dh1(x) = gradh1(x,nx,ny);
    dv1(x) = gradv1(x,nx,ny);
    dv2(x) = gradv2(x,nx,ny);

    Du = dv2(dh1(x));
    Dv = dv2(dv1(x));

    Dx = vcat(Du[:], Dv[:])./norme;
    return Dx;
end

function ComputeSltv7(x,nx,ny)

    #1er voisin diagonal gauche haut
    norme = 4.35;

    dh1(x) = gradh1(x,nx,ny);
    dv1(x) = gradv1(x,nx,ny);
    dh1v1(x) = gradh1v1haut(x,nx,ny);

    Du = dh1v1(dh1(x));
    Dv = dh1v1(dv1(x));

    Dx = vcat(Du[:], Dv[:])./norme;
    return Dx;
end

function ComputeSltv8(x,nx,ny)

    #1er voisin diagonal gauche bas
    norme = 4.35;

    dh1(x) = gradh1(x,nx,ny);
    dv1(x) = gradv1(x,nx,ny);
    dh1v1(x) = gradh1v1bas(x,nx,ny);

    Du = dh1v1(dh1(x));
    Dv = dh1v1(dv1(x));

    Dx = vcat(Du[:], Dv[:])./norme;
    return Dx;
end


function ComputeSltv_adj1(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);
    norme = 5.65;

    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);

    U = dh1_adj(dh1_adj(zh));
    V = dv1_adj(dh1_adj(zv));

    Dtz = (U[:] + V[:])./norme;
    return Dtz;
end

function ComputeSltv_adj2(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);

    # 2eme voisin horizontal gauche

    norme = 4.95;
    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);
    dh2_adj(x) = gradh2_adj(x,nx,ny);

    U = dh1_adj(dh2_adj(zh));
    V = dv1_adj(dh2_adj(zv));

    Dtz = (U[:] + V[:])./norme;
    return Dtz;
end

function ComputeSltv_adj4(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);

    # 1er voisin vertical haut

    norme = 5.65;
    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);

    U = dh1_adj(dv1_adj(zh));
    V = dv1_adj(dv1_adj(zv));

    Dtz = (U[:] + V[:])./norme;
    return Dtz;
end

function ComputeSltv_adj5(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);

    # 1er voisin vertical haut

    norme = 4.95;
    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);
    dv2_adj(x) = gradv2_adj(x,nx,ny);

    U = dh1_adj(dv2_adj(zh));
    V = dv1_adj(dv2_adj(zv));

    Dtz = (U[:] + V[:])./norme;
    return Dtz;
end

function ComputeSltv_adj7(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);

    #1er voisin diagonal gauche haut

    norme = 4.35;
    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);
    dh1v1_adj(x) = gradh1v1haut_adj(x,nx,ny);

    U = dh1_adj(dh1v1_adj(zh));
    V = dv1_adj(dh1v1_adj(zv));

    Dtz = (U[:] + V[:])./norme;
    return Dtz;
end

function ComputeSltv_adj8(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);

    # 1er voisin diagonal gauche bas

    norme = 4.35;
    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);
    dh1v1_adj(x) = gradh1v1bas_adj(x,nx,ny);

    U = dh1_adj(dh1v1_adj(zh));
    V = dv1_adj(dh1v1_adj(zv));

    Dtz = (U[:] + V[:])./norme;
    return Dtz;
end

# function prox_tvPrecond(coef, x, lambda, nx, ny, A)
#     x = sqrt(A).*x;
#     Ainv = sqrt(A).^(-1);

#     u = x[1:nx*ny];
#     v = x[nx*ny+1:end];

#     A1 = Ainv[1:nx*ny];
#     A2 = Ainv[nx*ny+1:end];
#     zu = zeros(nx*ny);
#     zv = copy(zu);

#     coeffu = coef*lambda* (A1.^2);
#     coeffv = coef*lambda* (A2.^2);

#     sqrtuv = sqrt((A1.*u).^2 + (A2.*v).^2);
#     #@printf("proxtv %d  %d  %d \n", nx, ny, length(sqrtuv))
#     indu = find(sqrtuv .> coeffu);
#     indv = find(sqrtuv .> coeffv);
#     zu[indu] = (1-coeffu[indu]./sqrtuv[indu]).*u[indu];
#     zv[indv] = (1-coeffv[indv]./sqrtuv[indv]).*v[indv];
#     z = Ainv.*vcat(zu, zv);
#     return z;
# end

function proxind(x,xmin,xmax)
    #p = min(max(x,xmin),xmax);
    return min.(max.(x,xmin),xmax);
end

function prox_l1(x, lambda)
    #p = max(abs(x)-lambda,0).*sign(x);
    return max.(abs.(x) .- lambda,0).*sign.(x);
end

function prox_tv(coef, x, lambda, nx, ny)
    u = x[1:nx*ny];
    v = x[nx*ny+1:end];
    zu = zeros(nx*ny);
    zv = copy(zu);
    coeff = coef*lambda;

    sqrtuv = sqrt.(u.^2 + v.^2);
    ind = findall(sqrtuv .> coeff);

    zu[ind] = (1 .- coeff./sqrtuv[ind]).*u[ind];
    zv[ind] = (1 .- coeff./sqrtuv[ind]).*v[ind];
    #z = vcat(zu, zv);
    return vcat(zu, zv);
end

function prox_hes(coef,x,lambda,nx,ny)
    u = x[1:nx*ny];
    v = x[nx*ny+1:2*nx*ny];
    y = x[2*nx*ny+1:end];

    zu = zeros(nx*ny);
    zv = copy(zu);
    zy = copy(zu);
    coeff = coef*lambda;

    sqrtuv = sqrt.(u.^2 + v.^2 + y.^2);
    ind = findall(sqrtuv .> coeff);

    zu[ind] = (1 .- coeff./sqrtuv[ind]).*u[ind];
    zv[ind] = (1 .- coeff./sqrtuv[ind]).*v[ind];
    zy[ind] = (1 .- coeff./sqrtuv[ind]).*y[ind];
    #z = vcat(zu, zv, zy);
    return vcat(zu, zv, zy);
end

function ComputeTVlin_alpha(x,nx,ny)
    Dx = vcat(gradh1(x, nx, ny)[:], gradv1(x,nx,ny)[:])./sqrt(8);
    return Dx;
end

function ComputeTVlinAdj_alpha(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);
    # U = gradh1_adj(zh,nx,ny);
    # V = gradv1_adj(zv,nx,ny);
    Dtz = (gradh1_adj(zh,nx,ny)[:] + gradv1_adj(zv,nx,ny)[:])./sqrt(8);
    return Dtz;
end

function ComputeTgv2(x,nx,ny,normeTgv2)
    x1 = reshape(x[1:nx*ny], nx,ny);
    x2 = reshape(x[nx*ny+1:end], nx,ny);

    dh1_adj(x) = gradh1_adj(x,nx,ny);
    dv1_adj(x) = gradv1_adj(x,nx,ny);
    d1 = dh1_adj(x1);
    d2 = dv1_adj(x1) + dh1_adj(x2);
    d3 = dv1_adj(x2);
    Dx = vcat(-d1[:], -d2[:], -d3[:])./normeTgv2;
    return Dx;
end

function ComputeTgv2Adj(z,nx,ny, normeTgv2)
    z1= reshape(z[1:nx*ny],nx,ny);
    z2= reshape(z[nx*ny+1:2*nx*ny],nx,ny);
    z3= reshape(z[2*nx*ny+1:end],nx,ny);

    #     dh1(x) = gradh1(x,nx,ny);
    #     dv1(x) = gradv1(x,nx,ny);
    #     dt1 = dh1(z1) + dv1(z2);
    #     dt2 = dh1(z2) + dv1(z3);
    Dtz = vcat(-gradh1(z1,nx,ny)[:] - gradv1(z2,nx,ny)[:], -gradh1(z2,nx,ny)[:]- gradv1(z3,nx,ny)[:])/normeTgv2;
    return Dtz;
end

function vectorize(x, n)
    p = [];
    for i in 1:n
        p = vcat(p, x[i][:]);
    end
    return p;
end

function ivectorize(x, n, nx, ny)
    p = Array{Array{Float32}}(undef, n);
    j = 1;
    for i in 1:n
        p[i] = reshape(x[j:j+ nx*ny-1], nx, ny);
        j += nx*ny;
    end
    return p;
end
