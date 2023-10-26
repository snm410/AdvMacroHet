import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r_m,r_a,w,vbeg_a_plus,vbeg_a,a,vbeg_m_plus,vbeg_m,m,c,l):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in nb.prange(par.Nz):
        
            ## i. labor supply
            l[i_fix,i_z,:] = par.z_grid[i_z]

            ## iia. cash-on-hand
            money_N = (1+r_m)*par.m_grid + w*l[i_fix,i_z,:]

            ## iia. cash-on-hand
            money_A = (1+r_a)*par.a_grid + (1+r_m)*par.m_grid - par.kappa + w*l[i_fix,i_z,:]

            # iiia. EGM
            c_endo_N = (par.beta_grid[i_fix]*vbeg_m_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo_N = c_endo_N + par.m_grid # current consumption + end-of-period assets
            
            # iiib. EGM
            c_endo_A = (par.beta_grid[i_fix]*vbeg_m_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo_A = c_endo_A + par.m_grid # current consumption + end-of-period assets


            # iv. interpolation to fixed grid
            interp_1d_vec(m_endo,par.m_grid,money,m[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            m[i_fix,i_z,:] = np.fmax(m[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = money-m[i_fix,i_z]

        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

        # # alternatively we write the matrix multiplication as a loop 
        # for i_z_lag in nb.prange(par.Nz):
        #     vbeg_a[i_fix,i_z_lag] = 0.0
        #     for i_z in range(par.Nz):
        #         vbeg_a[i_fix,i_z_lag] += z_trans[i_fix,i_z_lag,i_z]*v_a[i_fix,i_z]

