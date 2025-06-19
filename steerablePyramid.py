import sys
import os
import torch as t
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import datetime

'''
Based on "3D Steerable Pyramid based on conic filters" by
Celine A. Delle Luche, Florence Denis and Atilla Baskurt and
"A Parametric Texture Model Based on Joint Statistics
of Complex Wavelet Coefficients" by
Javier Portilla and Eero P. Simoncelli
'''

def LP( R, l, ref = 'PS' ) :
    # low-pass filter of order l in the Fourier space
    a = t.pi / 4
    b = t.pi / 2
    if ref == 'CS' :  # Simplified Design of Steerable Pyramid Filters", K. R. Castleman, M. Schulze, Q. Wu
        a /= 2**l
        b = 2 * a
        LP = t.sqrt( 1/2 * ( 1 + t.cos( t.pi * ( R - a ) / ( b - a ) ) ) )
        LP[ R <= a ] = 1
        LP[ R >= b ] = 0
    elif ref == 'PS' :  # "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" by Portilla and Simoncelli
        LP = t.cos( t.pi / 2 * t.log2( 4 * R * 2**l / t.pi ) )
        LP[ R * 2**l <= a ] = 1
        LP[ R * 2**l >= b ] = 0
    return LP

def HP( R, l, ref = 'PS' ) :
    # high-pass filter of order l in the Fourier space
    a = t.pi / 4
    b = t.pi / 2
    if ref == 'CS' :  # Simplified Design of Steerable Pyramid Filters", K. R. Castleman, M. Schulze, Q. Wu
        a /= 2**l
        b = 2 * a
        HP = t.sqrt( 1/2 * ( 1 - t.cos( t.pi * ( R - a ) / ( b - a ) ) ) )
        HP[ R >= b ] = 1
        HP[ R <= a ] = 0
    elif ref == 'PS' :  # "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" by Portilla and Simoncelli
        HP = t.cos( t.pi / 2 * t.log2( 2 * R * 2**l / t.pi ) )
        HP[ R * 2**l >= b ] = 1
        HP[ R * 2**l <= a ] = 0
    return HP

def OP( X, Y = None, Z = None, R = None, ndir = 1, m = 0, ref = 'PS' ) :
    # orientation-pass filter in the Fourier space for orientation m among ndir possible ones
    if Y is None :  # 1D input
        return 1.
    elif Z is None :  # 2D input
        # K. R. Castleman, M. A. Schulze, and Q. Wu, “Simplified design of steerable pyramid filters,”
        thm = t.tensor( t.pi * m / ndir )
        OP = ( X * t.cos( thm ) + Y * t.sin( thm ) ) / R  # cos of the angle between R and thm direction
        OP[ t.isnan( OP ) ] = 1.  # at the center
        if ref == 'PS' :
            # Javier Portilla and Eero P. Simoncelli in "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" suggested using asymmetric filter to create complex expansion components of an image
            OP[ OP < 0 ] = 0  # this makes the filter asymmetric (complex expansion components result)
        return OP**( ndir - 1 )
    else :  # 3D input
        # conical filter in direction m defined by a regular polyhedron with ndir 3D directioins
        p = ( 1 + t.sqrt( t.tensor( 5 ) ) ) / 2  # golden ratio
        if ndir == 3 :  # octahedron
            V = t.tensor( [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ] )
            n = 1
        elif ndir == 4 :  # cube
            V = t.tensor( [ [ 1, 1, 1 ], [ 1, -1, 1 ], [ -1, 1, 1 ], [ -1, -1, 1 ] ] ) / t.sqrt( t.tensor( 3 ) )
            n = 1 
        elif ndir == 6 :  # icosahedron
            V = t.tensor( [ [ p, 1, 0 ], [ p, -1, 0 ], [ 1, 0, p ], [ -1, 0, p ], [ 0, p, 1 ], [ 0, p, -1 ] ] ) / t.sqrt( p + 2 )
            n = 2
        elif ndir == 7 :  # rhombic dodecahedron (compound of octahedron and cube)
            VO = t.tensor( [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ] )
            VC = t.tensor( [ [ 1, 1, 1 ], [ 1, -1, 1 ], [ -1, 1, 1 ], [ -1, -1, 1 ] ] ) / t.sqrt( t.tensor( 3 ) )
            V = t.cat( [ VO, VC ], axis = 0 )
            n = 1
        elif ndir == 10 :  # dodecahedron
            V = t.tensor( [ [ 1, 1, 1 ], [ -1, 1, 1 ], [ 1, -1, 1 ], [ 1, 1, -1 ], [ 0, 1/p, p ], [ 0, -1/p, p ], [ 1/p, p, 0 ], [ -1/p, p, 0 ], [ p, 0, 1/p], [ p, 0, -1/p ] ] ) / t.sqrt( t.tensor( 3 ) )
            n = 2
        elif ndir == 16 :  # rhombic triacontahedron (compound of icosahedron and dodecahedron)
            VI = t.tensor( [ [ p, 1, 0 ], [ p, -1, 0 ], [ 1, 0, p ], [ -1, 0, p ], [ 0, p, 1 ], [ 0, p, -1 ] ] ) / t.sqrt( p + 2 )
            VD = t.tensor( [ [ 1, 1, 1 ], [ -1, 1, 1 ], [ 1, -1, 1 ], [ 1, 1, -1 ], [ 0, 1/p, p ], [ 0, -1/p, p ], [ 1/p, p, 0 ], [ -1/p, p, 0 ], [ p, 0, 1/p], [ p, 0, -1/p ] ] ) / t.sqrt( t.tensor( 3 ) )
            V = t.cat( [ VI, VD ], axis = 0 )
            n = 2
        else :
            print( 'ndir has to be one of the following: 3, 4, 6, 7, 10, 16!' )
            sys.exit()
        # V = t.sqrt( t.tensor( 3. / ndir ) ) * V  # normalize the conics so that all decomposition filters add up to 1
        # # plot directions used for the decomposition
        # fig = plt.figure()
        # ax = fig.add_subplot( projection='3d' )
        # ax.scatter( [ V[ :, 0 ], -V[ :, 0 ] ], [ V[ :, 1 ], -V[ :, 1 ] ], [ V[ :, 2 ], -V[ :, 2 ] ], marker = 'o' )
        # ax.set_aspect( 'equal' )
        OP = ( X * V[ m, 0 ] + Y * V[ m, 1 ] + Z * V[ m,2 ] ) / R
        OP[ t.isnan( OP ) ] = 1.  # at the center
        if ref == 'PS' :
            # Javier Portilla and Eero P. Simoncelli in "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" suggested using asymmetric filter to create complex expansion components of an image
            OP[ OP < 0 ] = 0  # this makes the filter asymmetric (complex expansion components result)
        return OP**n  # cosine of the angle between the polygon vertex m and the vector r

def steerablePyramid( dims, nsc, ndir, ref = 'PS' ) :
    #
    # returns steerable pyramid filters, ndir = 1 for 1D inputs
    #
    if ref != 'PS' and ref != 'CS' :
        print( "Only CS and PS options are allowed!" )
        sys.exit()

    ndims = len( dims )  # number of spatial dimensions (batch, channels, ... are not counted)
    if ndims == 1 :
        ndir = 1

    if ndims == 3 :
        X, Y, Z = t.meshgrid( t.linspace( -t.pi/2, t.pi/2, dims[ 0 ] ), t.linspace( -t.pi/2, t.pi/2, dims[ 1 ] ), t.linspace( -t.pi/2, t.pi/2, dims[ 2 ] ), indexing = 'xy' )
        X = X - X[ dims[ 0 ] // 2, dims[ 1 ] // 2, dims[ 2 ] // 2 ]  # the 1 px shift is necessary to align origin with t.fftn origin
        Y = Y - Y[ dims[ 0 ] // 2, dims[ 1 ] // 2, dims[ 2 ] // 2 ]
        Z = Z - Z[ dims[ 0 ] // 2, dims[ 1 ] // 2, dims[ 2 ] // 2 ]
        R = t.sqrt( X**2 + Y**2 + Z**2 )
    elif ndims == 2 :
        X, Y = t.meshgrid( t.linspace( -t.pi/2, t.pi/2, dims[ 0 ] ), t.linspace( -t.pi/2, t.pi/2, dims[ 1 ] ), indexing = 'xy' )
        X = X - X[ dims[ 0 ] // 2, dims[ 1 ] // 2 ]  # the 1 px shift is necessary to align origin with t.fftn origin
        Y = Y - Y[ dims[ 0 ] // 2, dims[ 1 ] // 2 ]
        Z = None
        R = t.sqrt( X**2 + Y**2 )
    elif ndims == 1 :
        X = t.linspace( -t.pi, t.pi, dims[ 0 ] )
        X = X - X[ dims[ 0 ] // 2 ]  # the 1 px shift is necessary to align origin with t.fftn origin
        Y = Z = None
        R = t.sqrt( X**2 )

    O = []  # orienation filters
    for d in range( ndir ) :  # loop over directions
        O.append( OP( X, Y, Z, R, ndir, d, ref ) )

    nfilts = 2 + nsc * ndir  # the total number of filters
    A = t.zeros( list( dims ) + [ nfilts ], requires_grad = False ) # all filters
    H = HP( R, 0, ref )  # zeroth-order high-pass
    L = LP( R, 0, ref )  # zeroth-order low-pass
    cnt = 0
    # A[ ..., cnt ] = H**2
    A[ ..., cnt ] = H
    for sc in range( 1, nsc + 1 ) :  # loop over frequency bands
        nrm = 0  # calculate the normalization of angular functions
        H = HP( R, sc, ref )
        for d in range( ndir ) :  # loop over directions
            nrm += ( O[ d ] * H )**2
        nrm = t.sqrt( t.max( nrm ) )
        if ref == 'PS' :  # need to increase the filter power by 2 to account for one lobe only
            nrm /= t.sqrt( t.tensor( 2. ) )
        B = H * L  # band-pass filter
        for d in range( ndir ) :  # loop over directions
            cnt += 1
            BO = O[ d ] / nrm * B  # band-pass oriented
            # A[ ..., cnt ] = t.abs( BO )**2
            A[ ..., cnt ] = BO
        L = LP( R, sc, ref ) * L  # the next-scale low-pass filter
    # A[ ..., cnt + 1 ] = L**2
    A[ ..., cnt + 1 ] = L

    return A

def expandImage( I, A, nsc, ndir, resto = False ) :
    #
    # expands I into nsc frequency bands and ndir orientation bands using filters A
    # if resto == True also returns partially restored images from the lowest ot the highest scale (wihtout the highest pass)
    #
    ndims = A.ndim - 1  # number of spatial dimensions (batch, channels, ... are not counted)
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    F = t.fft.fftshift( t.fft.fftn( I, dim = sdims ), dim = sdims )  # get the Fourier image centered with zero freq in the center
    Ic = [ 0 ] * A.shape[ -1 ]  # filtered image components
    A2 = A**2
    if resto :
        Ir = [ 0 ] * ( nsc + 1 )  # partially (up to scale i) reconstructed images
        A2sum = 0
    dims = np.array( A.shape[ : -1 ] )  # full size
    ctr = np.ceil( ( dims + 0.5 ) / 2 ).astype( int )
    # for i in range( A.shape[ -1 ] ) :
    for i in range( A.shape[ -1 ] - 1, -1, -1 ) :  # go from the lowest scale up
        if resto :
            A2sum = A2sum + A2[ ..., i ]
        # subsample components by cutting only the appropriate part of the Fourier image
        if i <= ndir :  # highpass and the first level bandpass-oriented
            sc = 1
        else :
            sc = 2**( ( i - 1 ) // ndir )
        lodims = np.ceil( ( dims - 0.5 ) / sc ).astype( int )
        loctr = np.ceil( ( lodims + 0.5 ) / 2 ).astype( int )
        sta = ctr - loctr
        fin = sta + lodims
        if ndims == 3 :
            Ic[ i ] = t.fft.ifftn( t.fft.ifftshift( A[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], sta[ 2 ] : fin[ 2 ], i ] * \
                                                    F[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], sta[ 2 ] : fin[ 2 ] ], dim = sdims ), dim = sdims )   # apply each filter in the Fourier space
            if resto and i % ndir == 1 :
                Ir[ i // ndir ] = t.real( t.fft.ifftn( t.fft.ifftshift( \
                A2sum[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], sta[ 2 ] : fin[ 2 ] ] * \
                    F[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], sta[ 2 ] : fin[ 2 ] ], dim = sdims ), dim = sdims ) )          
        elif ndims == 2 :
            Ic[ i ] = t.fft.ifftn( t.fft.ifftshift( A[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], i ] * \
                                                    F[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ] ], dim = sdims ), dim = sdims ) 
            if resto and i % ndir == 1 :
                Ir[ i // ndir ] = t.real( t.fft.ifftn( t.fft.ifftshift( \
                A2sum[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ] ] * \
                    F[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ] ], dim = sdims ), dim = sdims ) )
        elif ndims == 1 :
            Ic[ i ] = t.fft.ifftn( t.fft.ifftshift( A[ ..., sta[ 0 ] : fin[ 0 ], i ] * \
                                                    F[ ..., sta[ 0 ] : fin[ 0 ] ], dim = sdims ), dim = sdims )
            if resto and i > 0 :
                Ir[ i - 1 ] = t.real( t.fft.ifftn( t.fft.ifftshift( \
                A2sum[ ..., sta[ 0 ] : fin[ 0 ] ] * \
                    F[ ..., sta[ 0 ] : fin[ 0 ] ], dim = sdims ), dim = sdims ) )          
        # Ir[ ..., i // ndir ] = A2sum
    if resto :
        for i in range( len( Ir ) ) :
            Ir[ i ] = ( Ir[ i ] - t.mean( Ir[ i ] ) )  # / np.sqrt( 2 )
        return Ic, Ir
    else :
        return Ic

def restoreImage( Ic, A, nsc, ndir ) :
    #
    # restores I from nsc frequency bands and ndir orientation components Ic using filters A
    #
    ndims = A.ndim - 1  # number of spatial dimensions (batch, channels, ... are not counted)
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    dims = np.array( A.shape[ : -1 ] )  # full size
    F = t.zeros( tuple( dims ), dtype = complex )  # filtered image components
    ctr = np.ceil( ( dims + 0.5 ) / 2 ).astype( int )
    for i in range( A.shape[ -1 ] ) :
        if i <= ndir :  # highpass and the first level bandpass-oriented
            sc = 1
        else :
            sc = 2**( ( i - 1 ) // ndir )
        lodims = np.ceil( ( dims - 0.5 ) / sc ).astype( int )
        loctr = np.ceil( ( lodims + 0.5 ) / 2 ).astype( int )
        sta = ctr - loctr
        fin = sta + lodims
        Fc = t.fft.fftshift( t.fft.fftn( Ic[ i ], dim = sdims ), dim = sdims )  # get the Fourier image of the component
        if ndims == 3 :
            F[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], sta[ 2 ] : fin[ 2 ] ] += \
            A[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], sta[ 2 ] : fin[ 2 ], i ] * Fc
        elif ndims == 2 :
            F[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ] ] += \
            A[ ..., sta[ 0 ] : fin[ 0 ], sta[ 1 ] : fin[ 1 ], i ] * Fc
        elif ndims == 1 :
            F[ ..., sta[ 0 ] : fin[ 0 ] ] += \
            A[ ..., sta[ 0 ] : fin[ 0 ], i ] * Fc
    I = t.fft.ifftn( t.fft.ifftshift( F, dim = sdims ), dim = sdims )
    return I
    # return t.sum( Ic, dim = -1 )

def var( I, mean = None, ndims = None ) :
    # for arbitrary shaped tensors over the last ndims dimensions
    if ndims is None :
        ndims = I.ndim
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    if mean is None :
        if t.all( t.isreal( I ) ) :
            # need to explicitely broadcast mean here in case of images with batch and channels dimensions
            return t.mean( I**2, dim = sdims )
        else :
            return t.mean( t.real( I )**2, dim = sdims ) + \
              1j * t.mean( t.imag( I )**2, dim = sdims )
    else :
        if t.all( t.isreal( I ) ) :
            # need to explicitely broadcast mean here in case of images with batch and channels dimensions
            return t.mean( ( I - mean )**2, dim = sdims )
        else :
            return t.mean( t.real( I - mean )**2, dim = sdims ) + \
              1j * t.mean( t.imag( I - mean )**2, dim = sdims )

def skew( I, var, mean = None, ndims = None ) :
    # for arbitrary shaped tensors over the last ndim dimensions
    if ndims is None :
        ndims = I.ndim
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    if mean is None :
        if t.all( t.isreal( I ) ) :
            return t.mean( I**3, dim = sdims ) / var**(3/2)
        else :
            return t.mean( t.real( I )**3, dim = sdims ) / t.real( var )**(3/2) + \
              1j * t.mean( t.imag( I )**3, dim = sdims ) / t.imag( var )**(3/2)
    else :
        if t.all( t.isreal( I ) ) :
            return t.mean( ( I - mean )**3, dim = sdims ) / var**(3/2)
        else :
            return t.mean( t.real( I - mean )**3, dim = sdims ) / t.real( var )**(3/2) + \
              1j * t.mean( t.imag( I - mean )**3, dim = sdims ) / t.imag( var )**(3/2)

def kurt( I, var, mean = None, ndims = None ) :
    # for arbitrary shaped tensors over the last ndim dimensions
    if ndims is None :
        ndims = I.ndim
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    if mean is None :
        if t.all( t.isreal( I ) ) :
            return t.mean( I**4, dim = sdims ) / var**2
        else :
            return t.mean( t.real( I )**4, dim = sdims ) / t.real( var )**2 + \
             1j * t.mean( t.imag( I )**4, dim = sdims ) / t.imag( var )**2
    else :
        if t.all( t.isreal( I ) ) :
            return t.mean( ( I - mean )**4, dim = sdims ) / var**2
        else :
            return t.mean( t.real( I - mean )**4, dim = sdims ) / t.real( var )**2 + \
            1j * t.mean( t.imag( I - mean )**4, dim = sdims ) / t.imag( var )**2

def autocorrelate( I, la, ndims = None ) :
    # autocorrelation for arbitrary shaped tensors over the last ndim dimensions
    # returns the central (2*la+1)**ndim region
    if ndims is None :
        ndims = I.ndim
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    A = t.fft.fftshift( t.real( t.fft.ifftn( t.abs( t.fft.fftn( I, dim = sdims, norm = "ortho" ) )**2, dim = sdims ) ), dim = sdims )
    # A = t.fft.rfftn( I, dim = sdims )
    # A = A.real.pow( 2 ) + A.imag.pow( 2 )
    # A = t.fft.irfftn( A, dim = sdims )
    # A = t.fft.fftshift( A, dim = sdims ) / numel
    if ndims > 2 :
        ci = t.tensor( A.shape[ -3 ] // 2, dtype = t.int )
        lai = min( ci - 1, la )
        pi = la - lai
    if ndims > 1 :
        cj = t.tensor( A.shape[ -2 ] // 2, dtype = t.int )
        laj = min( cj - 1, la )
        pj = la - laj
    if ndims > 0 :
        ck = t.tensor( A.shape[ -1 ] // 2, dtype = t.int )
        lak = min( ck - 1, la )
        pk = la - lak
    # keep the central part, pad with zeros if the image is too small for the auto-correlation neighborhood la
    if ndims == 3 :
        var = A[ ..., ci, cj, ck ]  # the central element (variance) 
        A = A[ ..., ci - lai : ci + lai + 1, cj - laj : cj + laj + 1, ck - lak : ck + lak + 1 ] / var[ ..., None, None, None ] 
        return F.pad( A, [ pk, pk,  pj, pj,  pi, pi ], 'constant', 0 ), var  # padding proceeds from the last dim to the first
    elif ndims == 2 :
        var = A[ ..., cj, ck ]  
        A = A[ ..., cj - laj : cj + laj + 1, ck - lak : ck + lak + 1 ] / var[ ..., None, None ]
        return F.pad( A, [ pk, pk,  pj, pj ], 'constant', 0 ), var  # padding proceeds from the last dim to the first
    elif ndims == 1 :
        var = A[ ..., ck ]
        A = A[ ..., ck - lak : ck + lak + 1 ] / var[ ..., None ]  
        return F.pad( A, [ pk, pk ], 'constant', 0 ), var  # padding proceeds from the last dim to the first

def make_acorr_mask( nauto, ndims ) :
    # mask out identical entries given the inversion symmetry of the autocorrelation matrix
    h = nauto // 2
    M = np.zeros( [ nauto ] * ndims )
    if ndims == 3 :
        M[ :, :, 0 : h ] = 1  # all slices before the central
        M[ :, 0 : h, h ] = 1  # the central slice
        M[ 0 : h, h, h ] = 1  # the central row, the central element is not included because it is always 1
    elif ndims == 2 :
        M[ :, 0 : h ] = 1
        M[ 0 : h, h ] = 1  # the central element is not included because it is always 1
    elif ndims == 1 :
        M[ 0 : h ] = 1
    return t.tensor( M, dtype = t.bool )

def crosscorrelate( A, B, ndims, stdA = None, stdB = None ) :
    # cross-correlation for arbitrary shaped tensors over the [ -3, -2 ] dimensions
    # returns the Pearson correlation coefficient
    numel = t.prod( t.tensor( A.shape[ -ndims - 1 : -1 ] ) )
    # compute the covariance
    if ndims == 3 :
        dims = ' h w d '
    elif ndims == 2 :
        dims = ' h w '
    else :
        dims = ' h '
    covar = t.einsum( '...' + dims + 'a, ...' + dims + 'b -> ... a b', A, B ) / numel
    # normalize by standard deviations to get the Pearson product-moment correlation coefficient
    if stdA is None :
        stdA = t.sqrt( t.einsum( '...' + dims + 'a, ...' + dims + 'a -> ... a', A, A ) / numel )
    if stdB is None :
        stdB = t.sqrt( t.einsum( '...' + dims + 'a, ...' + dims + 'a -> ... a', B, B ) / numel )
    # outer product
    std_outer = t.einsum( "... a, ... b -> ... a b", stdA, stdB )
    return covar / std_outer

def get_image_statistics( I, nsc, ndir, nauto, Pyr, redundant = True ) :
    #
    # Returns pixel-level and intra-band / inter-band image statistics
    #
    ndims = Pyr.ndim - 1
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    if ndims == 1 :
        ndir = 1

    # Get pixel-level statistics
    # because torch doesn't support min and max over specific multiple dimensions one has to use reshape
    min0 = t.min( I.reshape( -1, t.prod( t.tensor( I.shape[ -ndims : ] ) ) ), dim = -1 )[ 0 ]
    min0 = min0.reshape( I.shape[ : -ndims ] )
    max0 = t.max( I.reshape( -1, t.prod( t.tensor( I.shape[ -ndims : ] ) ) ), dim = -1 )[ 0 ]
    max0 = max0.reshape( I.shape[ : -ndims ] )
    mean0 = t.mean( I, dim = sdims, keepdim = True )
    var0 = var( I, mean0, ndims )
    skew0 = skew( I, var0, mean0, ndims  )
    kurt0 = kurt( I, var0, mean0, ndims )
    mean0 = t.squeeze( mean0, dim = sdims )
    stats_vox = t.stack( [ mean0, var0, skew0, kurt0, min0, max0 ], dim = -1 )  # stack along new dimension

    # Build the steerable pyramid and return it along with partially restored images
    # st = datetime.datetime.now()
    Ic, Ir = expandImage( I, Pyr, nsc, ndir, resto = True )  # Ic - complex values, Ir - real values (partially restored)
    # d = datetime.datetime.now() - st
    # print( 'expand image:{}'.format( d ) )

    meanLF = t.mean( Ic[ -1 ], axis = sdims, keepdim = True )
    Ic[ -1 ] = Ic[ -1 ] - meanLF  # Subtract mean from the lowband

    # Compute central autocorrelation of the magnitude of all components except LP and HP
    nc = len( Ic )
    aIc = []
    mag_mean = t.zeros( tuple( np.append( I.shape[ : -ndims ], nc ).astype( int ) ) )
    for i in range( nc ) :
        aIc.append( t.abs( Ic[ i ] ) )  # magnitudes of the image components
        mn = t.mean( aIc[ i ], axis = sdims, keepdim = True )
        aIc[ i ] = aIc[ i ] - mn  # the last dimension stacks components 
        mag_mean[ ..., i ] = t.squeeze( mn, dim = sdims )
    acorr_mag = t.zeros( tuple( np.append( I.shape[ : -ndims ], [ nauto ] * ndims + [ nsc, ndir ] ).astype( int ) ) )
    mag_std = t.zeros( tuple( np.append( I.shape[ : -ndims ], [ nsc, ndir ] ).astype( int ) ) )  # to keep the variance (the central element of the autocorrelation matrix) given that it is normalized to 1    
    la = t.tensor( np.floor( ( nauto - 1) / 2 ).astype( int ) )
    # st = datetime.datetime.now()
    for i in range( nsc ) :
        for j in range( ndir ) :
            ind = 1 + i * ndir + j  # index of the component with scale i and direction j
            acorr_mag[ ..., i, j ], mag_var = autocorrelate( aIc[ ind ], la, ndims )
            mag_std[ ..., i, j ] = mag_var.sqrt()
    # d = datetime.datetime.now() - st
    # print( 'acorr_mag stats:{}'.format( d ) )

    # Compute central autocorrelation of the partially reconstructed image components at each scale (missing smaller-scale components)
    acorr_recon = t.zeros( tuple( np.append( I.shape[ : -ndims ], [ nauto ] * ndims + [ nsc + 1 ] ).astype( int ) ) )
    tmp = tuple( np.append( I.shape[ : -ndims ], nsc + 1 ).astype( int ) )
    skew_recon = t.zeros( tmp )  # +1 for the LP component, partially restored (except for smaller-scale bands)
    kurt_recon = t.zeros( tmp )
    std_recon = t.zeros( tmp )  # keep central elements of the autocorrelation matrices
    # st = datetime.datetime.now()
    for i in range( nsc + 1 ) :
        # if ndims == 3 :
        #     acorr_recon[ ..., :, :, :, i ], var_recon = autocorrelate( Ir[ i ], la, ndims )
        # elif ndims == 2 :
        #     acorr_recon[ ..., :, :, i ], var_recon = autocorrelate( Ir[ i ], la, ndims )
        # elif ndims == 1 :
        #     acorr_recon[ ..., :, i ], var_recon = autocorrelate( Ir[ i ], la, ndims )
        acorr_recon[ ..., i ], var_recon = autocorrelate( Ir[ i ], la, ndims )
        skew_recon[ ..., i ] = skew( Ir[ i ], var_recon, ndims = ndims )
        kurt_recon[ ..., i ] = kurt( Ir[ i ], var_recon, ndims = ndims )
        std_recon[ ..., i ]  = var_recon.sqrt()
    # d = datetime.datetime.now() - st
    # print( 'all recon stats:{}'.format( d ) )

    # Compute the cross-correlation matrices of the coefficient magnitudes at the same scale, the same for their real parts
    tmp = tuple( np.append( I.shape[ : -ndims ], [ int( ndir * ( ndir - 1 ) / 2 ), nsc ]).astype( int ) )
    xcorr_mag  = t.zeros( tmp )
    # st = datetime.datetime.now()
    R, C = np.triu_indices( ndir, 1 )  # indices for the upper triangular matrix
    for i in range( nsc ) :
        sta = 1 + i * ndir
        Is = t.zeros( tuple( np.append( aIc[ sta ].shape, ndir ).astype( int ) ) )
        for bi, bo in enumerate( range( sta, sta + ndir ) ) :  # collect all components at this scale
            Is[ ..., bi ] = aIc[ bo ]
        xcorr_mag[ ..., :, i ] = crosscorrelate( Is, Is, ndims, mag_std[ ..., i, : ], mag_std[ ..., i, : ] )[ ..., R, C ]
    # d = datetime.datetime.now() - st
    # print( 'xcorr_mag stats:{}'.format( d ) )

    # Compute the cross-correlation statistics with the lower scale
    if nsc > 1 :
        tmp = tuple( np.append( I.shape[ : -ndims ], [ ndir, ndir, nsc - 1 ]).astype( int ) )
        xcorr_lmag  = t.zeros( tmp )  # xcorr of magnitudes across all directions
        tmp = tuple( np.append( I.shape[ : -ndims ], [ ndir, 2 * ndir, nsc - 1 ]).astype( int ) )
        xcorr_lreim = t.zeros( tmp )  # xcorr of real parts with the lower-scale phase doubled first
        # st = datetime.datetime.now()
        for i in range( nsc - 1 ) :
            sta1 = 1 + i * ndir  # start of orientation bands for this scale
            fin1 = sta1 + ndir    # end of orientation bands
            sta2 = fin1
            fin2 = sta2 + ndir

            # collect all magnitude and real components at the current scale
            mI1 = t.zeros( tuple( np.append( aIc[ sta1 ].shape, ndir ).astype( int ) ) )
            rI1 = t.zeros_like( mI1 )
            for bi, bo in enumerate( range( sta1, fin1 ) ) :  # collect all components at this scale
                mI1[ ..., bi ] = aIc[ bo ]
                rI1[ ..., bi ] = t.real( Ic[ bo ] )

            # collect all compoenents at the lower scale
            Is2 = t.zeros( tuple( np.append( aIc[ sta2 ].shape, ndir ).astype( int ) ) )
            for bi, bo in enumerate( range( sta2, fin2 ) ) :  # collect all components at this scale
                Is2[ ..., bi ] = Ic[ bo ]

            # double the phases of the lower-scale components
            angle = Is2.angle()
            amp = Is2.abs()
            rIs = amp * t.cos( 2 * angle )
            iIs = amp * t.sin( 2 * angle )

            # upscale by 2
            rI2 = t.zeros_like( mI1 )
            iI2 = t.zeros_like( mI1 )
            if ndims == 3 :
                mode = 'trilinear'
            elif ndims == 2 :
                mode = 'bilinear'
            else :
                mode = 'linear'
            for bi in range( ndir ) :  # loop over components at this scale
                if rIs[ ..., bi ].ndim == ndims :  # no batch and channels
                    rI2[ ..., bi ] = 0.25 * t.squeeze( F.interpolate( rIs[ ..., bi ][ None, None, ... ], scale_factor = 2, mode = mode ), dim = ( 0, 1 ) )
                    iI2[ ..., bi ] = 0.25 * t.squeeze( F.interpolate( iIs[ ..., bi ][ None, None, ... ], scale_factor = 2, mode = mode ), dim = ( 0, 1 ) )
                else :
                    rI2[ ..., bi ] = 0.25 * F.interpolate( rIs[ ..., bi ], scale_factor = 2, mode = mode )
                    iI2[ ..., bi ] = 0.25 * F.interpolate( iIs[ ..., bi ], scale_factor = 2, mode = mode )

            xcorr_lmag[  ..., :, :, i ] = crosscorrelate( mI1, t.abs( rI2 + 1j * iI2 ), ndims, mag_std[ ..., i, : ] )  # mags cross-correlation
            xcorr_lreim[ ..., :, :, i ] = crosscorrelate( rI1, t.cat( [ rI2, iI2 ], dim = -1 ), ndims )  # real with real/imag cross-correlation
            # xcorr_lreim[ ..., :, :, i ] = crosscorrelate( rI1 + 1j * 0, rI2 + 1j * iI2, ndims )  # real with real/imag cross-correlation
        # d = datetime.datetime.now() - st
        # print( 'xcorr_lower stats:{}'.format( d ) )

    # Calculate the autocorrelation and variance of the HF residuals
    acorr_HP, var_HP = autocorrelate( t.real( Ic[ 0 ] ), la, ndims )

    if not redundant :  # remove redundant auto-correlation elements, form a single vector of statistics
        acorr_mask = make_acorr_mask( nauto, ndims )  # masks out equivalent (inverse-symmetric) elements of an auto-correlation matrix
        acorr_mag = acorr_mag[ ..., acorr_mask, :, : ].clone()
        acorr_recon = acorr_recon[ ..., acorr_mask, : ].clone()
        acorr_HP = acorr_HP[ ..., acorr_mask ]

    # 2 * mag_std is needed to match the plenoptic mag_std 
    stats_all = [ stats_vox, mag_mean, acorr_mag, skew_recon, kurt_recon, acorr_recon, std_recon, xcorr_mag, 2 * mag_std, acorr_HP, var_HP ]
    if nsc > 1 :
        # stats_all += [ 2 * xcorr_lmag, t.cat( [ t.real( xcorr_lreim ), t.imag( xcorr_lreim ) ], dim = -1 ) ]
        stats_all += [ 2 * xcorr_lmag, xcorr_lreim ]

    return stats_all

def vectorize( stats_all, min_level = None, weights = None ) :
    # collect all non-equivalent statistics into a vector for each image in a batch and the image channel
    got_weights = False
    if weights is not None :
        if t.is_tensor( weights ) and len( weights ) == len( stats_all ) :  # normalize by weights
            for si, s in enumerate( stats_all ) :
                stats_all[ si ] = s / weights[ si ]
        elif weights == True :  # calculate median values, one per category of statistcs
            got_weights = True
            weights = t.zeros( len( stats_all ) )
            for si, s in enumerate( stats_all ) :
                weights[ si ] = t.median( t.abs( s ) ) * t.numel( s )  # normalize by median value and number of elements in the block
                stats_all[ si ] /= weights[ si ]  # normalize

    # collect all non-equivalent statistics into a vector, auto-correlations in stats_all should be non-redundant!
    stats_vox, mag_mean, acorr_mag, skew_recon, kurt_recon, acorr_recon, std_recon, xcorr_mag, mag_std, acorr_HP, var_HP = stats_all[ : 11 ]
    if len( stats_all ) > 9 :
        xcorr_lmag, xcorr_lreim = stats_all[ 11 : ]
    else :
        xcorr_lmag = []
        xcorr_lreim = []

    vstats =  t.cat( [ stats_vox, mag_mean, \
                       t.flatten( acorr_mag[ ..., :, min_level :, : ], start_dim = -3, end_dim = -1 ), \
                       skew_recon[ ..., min_level : ], kurt_recon[ ..., min_level : ], \
                       t.flatten( acorr_recon[ ..., :, min_level : ], start_dim = -2, end_dim = -1 ), std_recon[ ..., min_level : ], \
                       t.flatten( xcorr_mag[ ..., :, min_level : ], start_dim = -2, end_dim = -1 ), \
                       t.flatten( mag_std[ ..., min_level :, : ], start_dim = -2, end_dim = -1 ), \
                       t.flatten( xcorr_lmag[ ..., :, :, min_level : ], start_dim = -3, end_dim = -1 ), \
                       t.flatten( xcorr_lreim[ ..., :, :, min_level : ], start_dim = -3, end_dim = -1 ), \
                        10 * acorr_HP, var_HP[ ..., None ] ], dim = -1 )
    if got_weights :
        return vstats, weights
    else :
        return vstats

        
if __name__ == "__main__" :
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # get available devices GPUs/CPU
    devices = t.device( "cuda" if t.cuda.is_available() else "cpu" )
    ndevices = t.cuda.device_count()
    print( 'Running on {} devices:'.format( ndevices ) )
    t.set_default_device( devices )  # set device for all new torch tensors

    nsc = 4  # number of spatial scales in the pyramid
    ndir = 4  # number of spatial directions in the pyramid
    nauto = 3  # span of the auto-correlation in each dimension

    # read 3D image
    sample = 'ref_textures/sawtooth3D_img.nii.gz'
    I = t.tensor( nib.load( sample ).get_fdata() )

    # # uncomment to test 2D image expansion
    # I = I[ :, :, 64 ].clone()  # make 2D to test 2D expansion

    # uncomment to test 1D signal expansion
    I = I[ :, 64, 64 ].clone()  # make 1D to test 1D expansion

    # I = t.tensor( plt.imread( 'ref_textures/straw.tiff' ) / 255., dtype = t.float )

    if I.ndim == 1 :
        ndir = 1

    # Get the set of image pyramid filters in the Fourier space. 
    # ref = CS or PS specifies Castellman or Portillo choice of filters.
    # The main difference is that Portillo and Simoncelli use asymmetric 
    # half-lobe oriented filters, while Castellman filters are symmetric. 
    Pyr = steerablePyramid( I.shape, nsc, ndir, ref = 'CS' )

    # get image statistics and put all non-equivalent statistics into a vector
    st = datetime.datetime.now()
    stats = get_image_statistics( I, nsc, ndir, nauto, Pyr, redundant = False )
    stats = vectorize( stats, min_level = None )
    d = datetime.datetime.now() - st
    print( 'Statistics calculation time:{} microseconds'.format( d.microseconds ) )
    print( 'Calculated {} statistics'.format( stats.shape[ 0 ] ) )

    # check that filters add up to 1 (only true for the CS filters)
    # s = t.sum( Pyr**2, axis = -1 )  # add up all filters
    s = t.sum( Pyr, axis = -1 )  # add up all filters
    print( 'Minimum filter sum value: {}'.format( t.min( s ) ) )
    print( 'Maximum filter sum value: {}'.format( t.max( s ) ) )

    # expand I into nsc frequency bands and ndir orientation bands
    # Ic = t.real( expandImage( I, Pyr ) )
    Ic = expandImage( I, Pyr, nsc, ndir )
    Ir = t.real( restoreImage( Ic, Pyr, nsc, ndir ) )
    # Ir = t.real( t.sum( Ic, axis = -1 ) )
    err = t.abs( Ir - I )
    print( 'Mean absolute error of the reconstruction: {}', format( t.mean( err ) ) )
    print( 'Max absolute error of the reconstruction: {}', format( t.max( err ) ) )

    # plot the expansion
    nfilts = len( Ic )
    fig, axs = plt.subplots( nsc, 1 + ndir )
    fig.set_figheight( 8 )
    fig.set_figwidth( 8 * np.max( [ 1, np.floor( ndir / nsc ).astype( int ) ] ) )
    mult = 1.  # multiplier to adjust image contrast limits
    if I.ndim == 3 :
        sl = I.shape[ 2 ] // 2  # draw this slice of the 3D volume
        img = I[ sl, ... ]
        Ir = Ir[ sl, ... ]
        imgc = []
        for i in range( nfilts ) :
            sl = Ic[ i ].shape[ 2 ] // 2  # draw this slice of the 3D volume
            imgc.append( Ic[ i ][ sl, ... ] )
        mult = 2.
    else :
        img = I
        imgc = Ic
    
    # get back on CPU for plotting
    img = img.detach().cpu()
    Ir = Ir.detach().cpu()

    if I.ndim > 1 :
        lc = 1
    plot_what = 'real'  # choose between plotting real/imag/abs values of the components
    for i in range( nfilts ) :
        if plot_what == 'abs' :
            imgc[ i ] = t.abs( imgc[ i ] )
            lc = 0  # set lower contrast limit to 0
        elif plot_what == 'real' :
            imgc[ i ] = t.real( imgc[ i ] )
        else :
            imgc[ i ] = t.imag( imgc[ i ] )

    cnt = 1
    for i in range( nsc ) :
        mult *= 2 * I.ndim / 2**( nsc - 1 - i )

        if i == 0 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( img )
            else :
                axs[ i, 0 ].imshow( img, vmin = .25, vmax = .75, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Original', fontsize=12 )
        elif i == 1 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( t.real( imgc[ -1 ] ) )
            else :
                axs[ i, 0 ].imshow( imgc[ -1 ], cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Lowest-pass', fontsize=12 )
        elif i == 2 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( t.real( imgc[ 0 ] ) )
            else :
                axs[ i, 0 ].imshow( imgc[ 0 ], cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Highest-pass', fontsize=12 )
        elif i == 3 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( Ir )
            else :
                axs[ i, 0 ].imshow( Ir, vmin = .25, vmax = .75, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Reconstruction', fontsize=12 )

        for j in range( 1, 1 + ndir ) :
            if I.ndim == 1 :
                axs[ i, j ].plot( t.real( imgc[ cnt ] ) )
            else :
                axs[ i, j ].imshow( imgc[ cnt ], vmin = -mult * lc, vmax = mult, cmap = 'gray' )
            axs[ i, j ].set_title( 'BP' + str( i + 1  ) +'Ori' + str( j ), fontsize=12 )
            cnt += 1

        for j in range( 1 + ndir ) :
            axs[ i, j ].set_xticks( [] )
            axs[ i, j ].set_yticks( [] )
    plt.show()