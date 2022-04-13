#include("prox_imageTGV_JMIVmertic.jl");
include("prox_imageTGV_PDC.jl");
function localOptimization(xstruct, imq, y, eta, alpha, alpha0, structMotion, BigBetaArrayI, N1_x, N2_x, xmin,xmax, gammaj, T, nb_frame, im_start, im_end,sequence)
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
        y_old = y[k];
        y_tild = y_old + gammaj* xstruct[k];
        q = imq[N1_x*N2_x*2*(k-1)+1: N1_x*N2_x*2*k];
        (proxi, imq[N1_x*N2_x*2*(k-1)+1: N1_x*N2_x*2*k]) = prox_imageTGV_PDC(gammaj^(-1)* y_tild, i, eta, alpha, N1_x, N2_x, xmin, xmax, sauv, nb, structMotion, BigBetaArrayI, alpha0, q, gammaj, T, sequence);
        y[k]  = y_tild - gammaj * proxi;
        xstruct[k] -= (y[k] - y_old);
    end

end
