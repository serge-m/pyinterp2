__author__ = 'Sergey Matyunin'

import numpy as np

# z = I1_
# s = np.array([0.5])
# t = np.array([0.5])
#z,XI,YI = I1_, idx,idy
#------------------------------------------------------
def interp2linear(z, XI, YI, extrapval=np.nan):
# if 1:
    x = XI.copy()
    y = YI.copy()
    nrows, ncols = z.shape;
    #     print "nrows,ncols", nrows,ncols

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")


    # find x values out of range
    x_bad = ( (x < 0) | (x > ncols - 1));
    #     print "x_bad", x_bad
    if x_bad.any():
    #         print "any!"
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1));
    #     print "y_bad", y_bad

    if y_bad.any():
    #         print "any!"
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x);
    ndx = ndx.astype('int32')
    #     print "ndx", ndx

    # fix parameters on x border
    #     if x.empty():
    #         d = x;
    #     else:
    #     print "x", x
    #     print "ncols-1", ncols-1
    d = (x == ncols - 1);
    x = (x - np.floor(x));
    if d.any():
        x[d] += 1;
        ndx[d] -= 1;
        #     print "ndx x", ndx

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y));
    if d.any():
        y[d] += 1;
        ndx[d] -= ncols;
        #     print "ndx y", ndx

    # interpolate
    one_minus_t = 1 - y;
    z = z.ravel()
    #     print "ndx", ndx
    #x, y = y, x
    #print "z[ndx]", z[ndx]
    #print "x", x
    #print "z[ndx]*(one_minus_t)", z[ndx]*(one_minus_t)
    #     print "z[ndx+1]*y", z[ndx+1]*y
    #     print "z[ndx+ncols]", z[ndx+ncols]
    #     print "ndx+(ncols)+1", ndx+(ncols)+1
    #     print "z[ndx+(ncols)+1]", z[ndx+(ncols)+1]
    F = ( z[ndx] * (one_minus_t) + z[ndx + ncols] * y ) * (1 - x) + ( z[ndx + 1] * (one_minus_t) + z[ndx + (ncols) + 1] * y ) * x;

    # Set out of range positions to extrapval
    if x_bad.any():
        F[x_bad] = extrapval;
    if y_bad.any():
        F[y_bad] = extrapval;


    return F
    # print F
    # z[ndx], z[ndx+1], z[ndx+ncols]
