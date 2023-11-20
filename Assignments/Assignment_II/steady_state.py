import time
import numpy as np
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):

        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    for i_fix in range(par.Nfix):
        
        # a. raw value
        ell = 1.0
        y = ss.wt*ell*par.z_grid
        c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
        v_a = (1+ss.r)*c**(-par.sigma)

        # b. expectation
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a

def obj_ss(KL,model,do_print=False):

    par = model.par
    ss = model.ss

    # a. firms
    ss.rK = par.alpha*par.Gamma_Y*(KL)**(par.alpha-1)
    ss.w = (1.0-par.alpha)*par.Gamma_Y*(KL)**par.alpha

    # b. arbitrage
    ss.r = ss.rK - par.delta

    # c. government
    ss.G = par.G_ss
    ss.L_G = par.L_G_ss
    ss.S = np.min((ss.G,par.Gamma_G*ss.L_G))
    ss.chi = par.chi_ss
    ss.tau = (ss.G + ss.w*ss.L_G + ss.chi)/(ss.w*ss.L_hh)
    print(f'tau = {ss.tau}')
    print(f'w = {ss.w}')

    # d. households
    ss.wt = (1-ss.tau)*ss.w

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    print(f'KL = {KL}')
    print(f'L_hh = {ss.L_hh}')

    # e. market clearing
    ss.B = ss.G + ss.w*ss.L_G + ss.chi - ss.tau*ss.w*ss.L_hh
    ss.L_Y = ss.L_hh - ss.L_G
    ss.K = KL*ss.L_Y
    ss.Y = par.Gamma_Y*ss.K**(par.alpha)*(ss.L_Y)**(1-par.alpha)
    ss.I = par.delta*ss.K
    ss.A = ss.K 
    ss.clearing_A = ss.A - ss.A_hh
    ss.clearing_L = ss.L_Y + ss.L_G - ss.L_hh
    ss.clearing_Y = ss.Y - ss.G - (ss.C_hh+ss.I)

    # Pi = ss.Y - ss.w*(ss.L_Y) - ss.rK*ss.K
    # print(Pi) 

    return ss.clearing_A

def find_ss(model,KL_min=None,KL_max=None,do_print=False):
    """ find the steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    if KL_min is None: KL_min = ((1/par.beta+par.delta-1)/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1)) + 1e-2 
    if KL_max is None: KL_max = (par.delta/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1))-1e-2

    # a. solve for K and L
    if do_print:
        print(f'seaching in [{KL_min:.4f},{KL_max:.4f}]')

    res = optimize.root_scalar(obj_ss,bracket=(KL_min,KL_max),method='brentq',args=(model,))
    
    # b. final evaluations
    obj_ss(res.root,model)

    # c. show
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.tau = :6.3f}')
        print(f'{ss.U_hh = :6.3f}')
        print(f'{ss.K = :6.3f}')
        print(f'{ss.B = :6.3f}')
        print(f'{ss.A_hh = :6.3f}')
        print(f'{ss.L_Y = :6.3f}')
        print(f'{ss.L_G = :6.3f}')
        print(f'{ss.L_hh = :6.3f}')
        print(f'{ss.G = :6.3f}')
        print(f'{ss.S = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.C_hh = :6.3f}')
        print(f'{ss.I = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.chi = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_L = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')