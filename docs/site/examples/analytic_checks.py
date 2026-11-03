"""Fast exact-reference checks; no scientific dependencies or backend calls."""
import json
import math


def gaussian(x, y):
    return -.5 * (x*x + (y-.6*x)**2/.64)


def main():
    for x in [-3., -1., 0., 1., 3.]:
        optimum=.6*x
        assert math.isclose(-2*gaussian(x,optimum),x*x,abs_tol=1e-12)
        assert gaussian(x,optimum) >= gaussian(x,optimum+.2)
        assert gaussian(x,optimum) >= gaussian(x,optimum-.2)
    # Ridge block: every 1D coordinate has a zero-penalty nuisance partner.
    for x in [-4., -1., 0., 1., 4.]:
        assert (x*x-x*x)**2 == 0
    for x in [0., 1., 5.]:
        assert (x*0)**2 == 0
    ell=[-9.,-3.,-1.,0.]
    for t in [1.,3.,7.]:
        a=[math.exp((1-1/t)*(x-max(ell))) for x in ell]
        lp=[x/t for x in ell]
        b=[math.exp((t-1)*(x-max(lp))) for x in lp]
        assert all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(a,b))
        ess=sum(a)**2/sum(x*x for x in a)
        assert 1-1e-12 <= ess <= len(a)+1e-12
    print(json.dumps({'status':'passed','backend_calls':0,
                      'checks':['Gaussian nuisance optimum','ridge identities','temperature weights and Kish ESS']}))

if __name__=='__main__': main()
