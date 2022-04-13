function prox_imageTGV_PDC(z, i, eta, alpha, N1_x, N2_x, xmin, xmax, sauv, nb, structMotion, BigBetaArrayI, alpha0, q, gammaj, T, sequence)
    #Primal-Dual algorithm
    prec = 1e-5;
    NbrIt = 30; #100

    if(alpha>0)
        betaArrayI = BigBetaArrayI[i];
        motionTab = structMotion[i]["Motion"];
        var = matread("$(sequence)/Motion/NormXtXm$i.mat");
        normL = copy(var["normL"]);
    else
        betaArrayI = 0;
        motionTab = 0;
        normL = 1.617;
    end

    gamma1 = 1e-1;
    gamma2 = (1/gamma1 - 1/2)/(normL^2);

    normTgv2 = 3.4544;

    L2(x) = ComputeTVlin_alpha(x,N1_x,N2_x);
    L2_adj(z) = ComputeTVlinAdj_alpha(z,N1_x,N2_x);

    L3(x) = ComputeTgv2(x,N1_x,N2_x, normTgv2);
    L3_adj(z) = ComputeTgv2Adj(z,N1_x,N2_x,normTgv2);

    LMotionForward(xt, xl, mat) = xt - mat*xl;
    LMotionForward_adj(x, mat) = vcat(x, - mat'*x);

    x = copy(z);

    qnew = zeros(2*N1_x*N2_x);
    z1 = zeros(2*N1_x*N2_x);
    z2 = zeros(3*N1_x*N2_x);

    w1 = zeros(2*N1_x*N2_x);
    w2 = zeros(3*N1_x*N2_x);
    z_m = Array{Any}(undef,sauv+nb);
    w_m = Array{Any}(undef,sauv+nb);
    for k in 1:sauv+nb
        z_m[k] = zeros(N1_x*N2_x);
        w_m[k] = zeros(N1_x*N2_x);
    end

    alpha1 = 1-alpha0;

    gradF = zeros(N1_x*N2_x*(1+sauv+nb));
    critO = 0;

    for k in 1:NbrIt
        res= zeros(N1_x*N2_x*(1+sauv+nb));
        gradF = x - z;
        gradF[1:N1_x*N2_x]*= gammaj*1/(1+sauv+nb);
        for l in 1:sauv
            xtemp = LMotionForward_adj(z_m[l], motionTab[l]);
            res[1:N1_x*N2_x]+=xtemp[1:N1_x*N2_x];
            res[N1_x*N2_x*l+1: N1_x*N2_x*(l+1)] += xtemp[N1_x*N2_x+1: end];
            other_image = i-sauv+l-1;
            if(other_image ==1)
                wj =1/2;
            else
                wj = 1/3;
            end
            gradF[N1_x*N2_x*l+1: N1_x*N2_x*(l+1)]*= gammaj*wj;
        end

        for l in 1:nb
            xtemp = LMotionForward_adj(z_m[l+sauv], motionTab[l+sauv]);
            res[1:N1_x*N2_x]+=xtemp[1:N1_x*N2_x];
            res[N1_x*N2_x*(l+sauv)+1: N1_x*N2_x*(l+sauv+1)] += xtemp[N1_x*N2_x+1: end];
            other_image = i+nb-l+1;
            if(other_image==T)
                wj = 1/2;
            else
                wj = 1/3;
            end
            gradF[N1_x*N2_x*(l+sauv)+1: N1_x*N2_x*(l+sauv+1)]*= gammaj*wj;
        end
        
        xnew = proxind(x - gamma1*(gradF + [L2_adj(z1);zeros(N1_x*N2_x*(sauv+nb))] +res), xmin, xmax);
        qnew = q - gamma1*(-z1 + L3_adj(z2));

        w1 = z1 + gamma2*(L2(2*xnew[1:N1_x*N2_x] - x[1:N1_x*N2_x]) - (2*qnew - q));
        w2 = z2 + gamma2*L3(2*qnew - q);

        for l in 1:sauv+nb
            w_m[l] = z_m[l] + gamma2* LMotionForward(2*xnew[1:N1_x*N2_x] - x[1:N1_x*N2_x], 2*xnew[N1_x*N2_x*l+1: N1_x*N2_x*(l+1)] - x[N1_x*N2_x*l+1: N1_x*N2_x*(l+1)], motionTab[l]);
        end

        z1 = w1 - gamma2*prox_tv(eta*alpha1*sqrt(8), w1/gamma2, 1/gamma2, N1_x, N2_x);
        z2 = w2 - gamma2*prox_hes(eta*alpha0*normTgv2, w2/gamma2, 1/gamma2, N1_x, N2_x);

        for l in 1:sauv+nb
            z_m[l] = w_m[l] - gamma2*prox_l1(w_m[l]/gamma2, alpha*betaArrayI[2,l]/gamma2);
        end

        x = copy(xnew);
        q = copy(qnew);
    end
    return x, q;

end
#----------------------------------------------------------------------------------------
#                                Functions
#---------------------------------------------------------------------------------------

function ComputeTVlin_alpha(x,nx,ny)
    Dx = vcat(gradh1(x, nx, ny)[:], gradv1(x,nx,ny)[:])./sqrt(8);
    return Dx;
end

function ComputeTVlinAdj_alpha(z,nx,ny)
    zh= reshape(z[1:nx*ny],nx,ny);
    zv= reshape(z[nx*ny+1:end],nx,ny);
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

    Dtz = vcat(-gradh1(z1,nx,ny)[:] - gradv1(z2,nx,ny)[:], -gradh1(z2,nx,ny)[:]- gradv1(z3,nx,ny)[:])/normeTgv2;
    return Dtz;
end

function critereTGVIn(x, z, q, i, eta, alpha, sauv, nb, N1_x, N2_x, BigBetaArrayI, structMotion, alpha0)

    crit = sum((x- z).^2);
    alpha1 = 1-alpha0;

    xtemp = x[1:N1_x*N2_x];

    Rx1 = alpha1 *ComputeTVq(reshape(xtemp,N1_x, N2_x), q, N1_x,N2_x);
    Rx2 = alpha0 * ComputeTGV(q,N1_x,N2_x);
    crit += eta*(Rx1+Rx2);

    if(alpha >0)
        betaArrayI = BigBetaArrayI[i];
        motionTab = structMotion[i]["Motion"];

        for l = 1:sauv+nb
            r = x[N1_x*N2_x*l+ 1: N1_x*N2_x*(l+1)];
            crit += alpha* betaArrayI[2,l]*(sum(abs(xtemp - motionTab[l]*r)));
        end
    end
    return crit;
end
