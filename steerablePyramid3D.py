import sys
import torch as t
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

'''
Based on "3D Steerable Pyramid based on conic filters" by
C´eline A. Delle Luche, Florence Denis and Atilla Baskurt and
"A Parametric Texture Model Based on Joint Statistics
of Complex Wavelet Coefficients" by
Javier Portilla and Eero P. Simoncelli
'''

def LP( R, l ) :
    # low-pass filter of order l in the interval [ 0, k ]
    a = t.pi / 4 / ( l + 1 )
    b = 2 * a
    LP = t.sqrt( 1/2 * ( 1 + t.cos( t.pi * ( R - a ) / ( b - a ) ) ) )
    LP[ R >= b ] = 0
    LP[ R <= a ] = 1
    return LP

def HP( R, l ) :
    # high-pass filter of order l in the interval [ 0, k ]
    a = t.pi / 4 / ( l + 1 )
    b = 2 * a
    HP = t.sqrt( 1/2 * ( 1 - t.cos( t.pi * ( R - a ) / ( b - a ) ) ) )
    HP[ R >= b ] = 1
    HP[ R <= a ] = 0
    return HP

def OP( X, Y = None, Z = None, R = None, ndir = 1, m = 0 ) :
    if Y is None :  # 1D input
        return 1.
    elif Z is None :  # 2D input
        thm = t.tensor( t.pi * m / ndir )
        return ( ( X * t.cos( thm ) + Y * t.sin( thm ) ) / R )**( ndir - 1 ) 
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
        return ( ( X * V[ m, 0 ] + Y * V[ m, 1 ] + Z * V[ m,2 ] ) / R )**n  # cosine of the angle between the polygon vertex m and the vector r

def steerablePyramid( dims, nsc, ndir ) :
    #
    # returns steerable pyramid filters, ndir = 1 for 1D inputs
    #
    ndims = len( dims )  # number of spatial dimensions (batch, channels, ... are not counted)
    if ndims == 1 :
        ndir = 1

    if ndims == 3 :
        X, Y, Z = t.meshgrid( t.linspace( -t.pi, t.pi, dims[ 0 ] ), t.linspace( -t.pi, t.pi, dims[ 1 ] ), t.linspace( -t.pi, t.pi, dims[ 2 ] ), indexing = 'ij' )
        R = t.sqrt( X**2 + Y**2 + Z**2 )
    elif ndims == 2 :
        X, Y = t.meshgrid( t.linspace( -t.pi, t.pi, dims[ 0 ] ), t.linspace( -t.pi, t.pi, dims[ 1 ] ), indexing = 'ij' )
        Z = None
        R = t.sqrt( X**2 + Y**2 )
    elif ndims == 1 :
        X = t.linspace( -t.pi, t.pi, dims[ 0 ] )
        Y = Z = None
        R = t.sqrt( X**2 )

    O = []  # orienation filters
    for d in range( ndir ) :  # loop over directions
        O.append( OP( X, Y, Z, R, ndir, d ) )

    nfilts = 2 + nsc * ndir  # the total number of filters
    A2 = t.zeros( list( dims ) + [ nfilts ], requires_grad = False ) # all filters squared
    H = HP( R, 0 )  # zeroth-order high-pass
    L = LP( R, 0 )  # zeroth-order low-pass
    cnt = 0
    A2[ ..., cnt ] = H**2
    for sc in range( 1, nsc + 1 ) :  # loop over frequency bands
        nrm = 0  # calculate normalization of angular functions so that they sum up to 1 with the next low-pass**2
        for d in range( ndir ) :  # loop over directions
            nrm += ( O[ d ] * HP( R, sc ) )**2
        nrm = t.sqrt( t.max( nrm ) )
        b = HP( R, sc ) * L  # band-pass filter
        for d in range( ndir ) :  # loop over directions
            cnt += 1
            BO = O[ d ] / nrm * b  # band-pass oriented
            A2[ ..., cnt ] = t.abs( BO )**2
        L = LP( R, sc ) * L  # the next-scale low-pass filter
    A2[ ..., cnt + 1 ] = L**2

    return A2

def expandImage( I, A2 ) :
    #
    # expands I into nsc frequency bands and ndir orientation bands
    #
    ndims = A2.ndim - 1  # number of spatial dimensions (batch, channels, ... are not counted)
    sdims = list( range( -ndims, 0 ) )  # spatial dims in an image
    F = t.fft.fftshift( t.fft.fftn( I, dim = sdims ), dim = sdims )  # get the Fourier image centered with zero freq in the center
    Ic = t.zeros( tuple( np.append( I.shape, A2.shape[ -1 ] ) ), dtype = complex )  # filtered image components
    for i in range( A2.shape[ -1 ] ) :
        Ic[ ..., i ] = t.fft.ifftn( t.fft.ifftshift( A2[ ..., i ] * F, dim = sdims ), dim = sdims )   # apply each filter in the Fourier space
    return Ic

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
    A = t.fft.fftshift( t.real( t.fft.ifftn( t.abs( t.fft.fftn( I, dim = sdims ) )**2, dim = sdims ) ), dim = sdims ) / t.prod( t.tensor( I.shape ) )
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
        A = A[ ..., ci - lai : ci + lai + 1, cj - laj : cj + laj + 1, ck - lak : ck + lak + 1 ]    
        return F.pad( A, [ pk, pk,  pj, pj,  pi, pi ], 'constant', 0 )  # padding proceeds from the last dim to the first
    elif ndims == 2 :
        A = A[ ..., cj - laj : cj + laj + 1, ck - lak : ck + lak + 1 ]    
        return F.pad( A, [ pk, pk,  pj, pj ], 'constant', 0 )  # padding proceeds from the last dim to the first
    elif ndims == 1 :
        A = A[ ..., ck - lak : ck + lak + 1 ]    
        return F.pad( A, [ pk, pk ], 'constant', 0 )  # padding proceeds from the last dim to the first

def make_acorr_mask( nauto, ndims ) :
    # mask identical entries given the inversion symmetry of the autocorrelation matrix
    h = nauto // 2
    M = np.zeros( [ nauto ] * ndims )
    if ndims == 3 :
        M[ 0 : nauto, 0 : nauto, h : nauto ] = 1
        M[ 0 : nauto, 0 : h, h ] = 0  # remove the smaller half of the central slice 
        M[ 0 : h, h, h ] = 0  # remove the smaller half of the central x axis
    elif ndims == 2 :
        M[ 0 : nauto, h : nauto ] = 1
        M[ 0 : h, h ] = 0
    elif ndims == 1 :
        M[ h : nauto ] = 1
    return t.tensor( M, dtype = t.bool )

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

    # Build the steerable pyramid
    Ic = expandImage( I, Pyr )  # complex values
    meanLF = t.mean( Ic[ ..., -1 ], axis = sdims, keepdim = True )
    Ic[ ..., -1 ] = Ic[ ..., -1 ] - meanLF  # Subtract mean from the lowband
    # Calculate the variance of the HF residuals
    varHF = t.mean( t.real( Ic[ ..., 0 ] )**2, dim = sdims )  # the imag part is negligible, within roundoff errors here, as it should
    stats_vox = t.stack( [ mean0, var0, skew0, kurt0, min0, max0, varHF ], dim = -1 )  # stack along new dimension

    # Compute central autocorrelation of the magnitude of all components except LP and HP
    aIc = t.abs( Ic )  # magnitudes of the image components
    sdims_here = tuple( np.array( sdims ) - 1 )
    mag_means = t.mean( aIc, axis = sdims_here, keepdim = True )  # the last dimension stacks components 
    aIc = aIc - mag_means # Subtract mean from the magnitude
    mag_means = t.squeeze( mag_means, dim = sdims_here )
    acorr_mag = t.zeros( tuple( np.append( I.shape[ : -ndims ], [ nauto ] * ndims + [ nsc, ndir ] ).astype( int ) ) )
    la = t.tensor( np.floor( ( nauto - 1) / 2 ).astype( int ) )
    for i in range( nsc ) :
        s = 2**i  # subsampling step
        for j in range( ndir ) :
            ind = 1 + i * ndir + j  # index of the component with scale i and diraction j
            if ndims == 3 :
                Is = aIc[ ..., 0::s, 0::s, 0::s, ind ].clone()  # the subsampled component
                acorr_mag[ ..., :, :, :, i, j ] = autocorrelate( Is, la, ndims )
            elif ndims == 2 :
                Is = aIc[ ..., 0::s, 0::s, ind ].clone()  # the subsampled component
                acorr_mag[ ..., :, :, i, j ] = autocorrelate( Is, la, ndims )
            elif ndims == 1 :
                Is = aIc[ ..., 0::s, ind ].clone()  # the subsampled component
                acorr_mag[ ..., :, i, j ] = autocorrelate( Is, la, ndims )

    # Compute central autocorrelation of the partially reconstructed image components at each scale (missing smaller-scale components)
    rIc = t.real( Ic )
    acorr_recon = t.zeros( tuple( np.append( I.shape[ : -ndims ], [ nauto ] * ndims + [ nsc + 1 ] ).astype( int ) ) )
    tmp = tuple( np.append( I.shape[ : -ndims ], nsc + 1 ).astype( int ) )
    skew_recon = t.zeros( tmp )  # +1 for the LP component, partially restored (except for smaller-scale bands)
    kurt_recon = t.zeros( tmp )
    for i in range( nsc + 1 ) :
        s = 2**i  # subsampling step
        if ndims == 3 :
            Is = rIc[ ..., 0::s, 0::s, 0::s, -1 ].clone()  # the lowest-pass component subsampled for this band
            if i < nsc :  # add all orientations in this band subsampled
                Is += t.sum( rIc[ ..., 0::s, 0::s, 0::s, 1 + i * ndir : 1 + ( i + 1 ) * ndir ], axis = -1 )  # Is has mean = 0
            acorr_recon[ ..., :, :, :, i ] = autocorrelate( Is, la, ndims )
            var_recon = acorr_recon[ ..., la, la, la, i ].clone()  # central element of the auto-correlation matrix
        elif ndims == 2 :
            Is = rIc[ ..., 0::s, 0::s, -1 ].clone()  # the lowest-pass component subsampled for this band
            if i < nsc :  # add all orientations in this band subsampled
                Is += t.sum( rIc[ ..., 0::s, 0::s, 1 + i * ndir : 1 + ( i + 1 ) * ndir ], axis = -1 )  # Is has mean = 0
            acorr_recon[ ..., :, :, i ] = autocorrelate( Is, la, ndims )
            var_recon = acorr_recon[ ..., la, la, i ].clone()  # central element of the auto-correlation matrix
        elif ndims == 1 :
            Is = rIc[ ..., 0::s, -1 ].clone()  # the lowest-pass component subsampled for this band
            if i < nsc :  # add all orientations in this band subsampled
                Is += t.sum( rIc[ ..., 0::s, 1 + i * ndir : 1 + ( i + 1 ) * ndir ], axis = -1 )  # Is has mean = 0
            acorr_recon[ ..., :, i ] = autocorrelate( Is, la, ndims )
            var_recon = acorr_recon[ ..., la, i ].clone()  # central element of the auto-correlation matrix
        skew_recon[ ..., i ] = skew( Is, var_recon, ndims = ndims )
        kurt_recon[ ..., i ] = kurt( Is, var_recon, ndims = ndims )

    # Compute the cross-correlation matrices of the coefficient magnitudes at the same scale, the same for their real parts
    tmp = tuple( np.append( I.shape[ : -ndims ], [ int( ndir * ( ndir - 1 ) / 2 ), nsc ]).astype( int ) )
    xcorr_mag  = t.zeros( tmp )
    xcorr_real = t.zeros( tmp )
    for i in range( nsc ) :
        cnt = 0
        for j in range( ndir ) :
            b1 = 1 + i * ndir + j
            for k in range( j + 1, ndir ) :
                b2 = 1 + i * ndir + k
                xcorr_mag[ ..., cnt, i ]  = t.mean( aIc[ ..., b1 ] * aIc[ ..., b2 ], dim = sdims )  # xcorr between mags of orientations j and k
                xcorr_real[ ..., cnt, i ] = t.mean( rIc[ ..., b1 ] * rIc[ ..., b2 ], dim = sdims )  # the same for real parts
                cnt += 1

    # Compute the cross-correlation statistics with the lower scale
    tmp = tuple( np.append( I.shape[ : -ndims ], [ ndir, ndir, nsc - 1 ]).astype( int ) )
    xcorr_lmag  = t.zeros( tmp )  # xcorr of magnitudes across all directions
    xcorr_lreal = t.zeros( tmp )  # xcorr of real parts with the lower-scale phase doubled first
    xcorr_limag = t.zeros( tmp )  # xcorr of real and imaginary parts with the lower-scale phase doubled first
    for i in range( nsc - 1 ) :
        for j in range( ndir ) :
            b1 = 1 + i * ndir + j  # current-scale component with orientation j
            for k in range( ndir ) :
                b2 = 1 + ( i + 1 ) * ndir + k  # lower-scale component of orientation k
                xcorr_lmag[ ..., j, k, i ] = t.mean( aIc[ ..., b1 ] * aIc[ ..., b2 ], dim = sdims )  # xcorr between mags of orientations j and k
                # double the phases of the lower-scale component
                Is = t.abs( Ic[ ..., b2 ] ) * t.exp( 2 * 1j * t.angle( Ic[ ..., b2 ] ) )
                xcorr_lreal[ ..., j, k, i ] = t.mean( rIc[ ..., b1 ] * t.real( Is ), dim = sdims )
                xcorr_limag[ ..., j, k, i ] = t.mean( rIc[ ..., b1 ] * t.imag( Is ), dim = sdims )

    if not redundant :  # remove redundant auto-correlation elements, form a single vector of statistics
        acorr_mask = make_acorr_mask( nauto, ndims )  # masks out equivalent (inverse-symmetric) elements of an auto-correlation matrix
        acorr_mag = acorr_mag[ ..., acorr_mask, :, : ].clone()
        acorr_recon = acorr_recon[ ..., acorr_mask, : ].clone()

    return [ stats_vox, mag_means, skew_recon, kurt_recon, acorr_recon, acorr_mag, xcorr_mag, xcorr_real, xcorr_lmag, xcorr_lreal, xcorr_limag ]

def vectorize( stats_all, min_level = None, weights = None ) :
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
    stats_vox, mag_means, skew_recon, kurt_recon, acorr_recon, acorr_mag, xcorr_mag, xcorr_real, xcorr_lmag, xcorr_lreal, xcorr_limag, = stats_all

    ndirs = acorr_mag.shape[ -1 ]  # number of scales including the lowest-pass scale
    if min_level is None :  # include all statistics in the pyramid
        min_level = 0 
        stats = t.cat( [ stats_vox, mag_means ], dim = -1 )
    else :
        stats = mag_means[ ..., 1 + min_level * ndirs : ]
    vstats =  t.cat( [ stats, skew_recon[ ..., min_level : ], kurt_recon[ ..., min_level : ], \
                        t.flatten( acorr_recon[ ..., :, min_level : ], start_dim = -2, end_dim = -1 ), \
                        t.flatten( acorr_mag[ ..., :, min_level :, : ], start_dim = -3, end_dim = -1 ), \
                        t.flatten( xcorr_mag[ ..., :, min_level : ], start_dim = -2, end_dim = -1 ), \
                        t.flatten( xcorr_real[ ..., :, min_level : ], start_dim = -2, end_dim = -1 ), \
                        t.flatten( xcorr_lmag[ ..., :, :, min_level : ], start_dim = -3, end_dim = -1 ), \
                        t.flatten( xcorr_lreal[ ..., :, :, min_level : ], start_dim = -3, end_dim = -1 ), \
                        t.flatten( xcorr_limag[ ..., :, :, min_level : ], start_dim = -3, end_dim = -1 ) ], dim = -1 )
    if got_weights :
        return vstats, weights
    else :
        return vstats

        
if __name__ == "__main__" :
    nsc = 4  # number of spatial scales in the pyramid
    ndir = 4  # number of spatial directions in the pyramid
    nauto = 5  # span of the auto-correlation in each dimension

    # read 3D image
    sample = 'sawtooth3D_img.nii.gz'
    I = t.tensor( nib.load( sample ).get_fdata() )

    # # uncomment to test 2D image expansion
    # I = I[ :, :, 32 ].clone()  # make 2D to test 2D expansion

    # uncomment to test 1D signal expansion
    I = I[ :, 32, 32 ].clone()  # make 1D to test 1D expansion

    if I.ndim == 1 :
        ndir = 1

    # get image pyramid expansion
    Pyr = steerablePyramid( I.shape, nsc, ndir )

    # get image statistics and put all non-equivalent statistics into a vector
    stats_all = get_image_statistics( I, nsc, ndir, nauto, Pyr, redundant = False )
    stats_all = vectorize( stats_all, min_level = None )
    print( 'Calculated {} statistics'.format( stats_all.shape[ 0 ] ) )

    # check that filters add up to 1
    s = t.sum( Pyr, axis = -1 )  # add up all filters
    print( 'Minimum filter sum value: {}'.format( t.min( s ) ) )
    print( 'Maximum filter sum value: {}'.format( t.max( s ) ) )

    # expand I into nsc frequency bands and ndir orientation bands
    Ic = t.real( expandImage( I, Pyr ) )
    res = t.sum( Ic, axis = -1 )
    err = t.abs( res - I )
    print( 'Mean absolute error of the reconstruction: {}', format( t.mean( err ) ) )
    print( 'Max absolute error of the reconstruction: {}', format( t.max( err ) ) )

    # plot the expansion
    nfilts = Ic.shape[ -1 ]
    fig, axs = plt.subplots( nsc, 1 + ndir )
    fig.set_figheight( 8 )
    fig.set_figwidth( 8 * np.max( [ 1, np.floor( ndir / nsc ).astype( int ) ] ) )
    if I.ndim == 3 :
        sl = I.shape[ 2 ] // 2  # draw this slice of the 3D volume
        img = I[ sl, ... ]
        imgc = Ic[ sl, ... ]
    else :
        img = I
        imgc = Ic
    cnt = 1
    for i in range( nsc ) :
        if i == 0 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( img )
            else :
                axs[ i, 0 ].imshow( img, vmin = .25, vmax = .75, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Original', fontsize=12 )
        elif i == 1 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( imgc[ ..., -1 ] )
            else :
                axs[ i, 0 ].imshow( imgc[ ..., -1 ], vmin = 0, vmax = 1, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Lowest-pass', fontsize=12 )
        elif i == 2 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( imgc[ ..., 0 ] )
            else :
                axs[ i, 0 ].imshow( imgc[ ..., 0 ], vmin = 0, vmax = .2, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Highest-pass', fontsize=12 )
        elif i == 3 :
            if I.ndim == 1 :
                axs[ i, 0 ].plot( t.sum( imgc, axis = -1 ) )
            else :
                axs[ i, 0 ].imshow( t.sum( imgc, axis = -1 ), vmin = .25, vmax = .75, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Reconstruction', fontsize=12 )

        for j in range( 1, 1 + ndir ) :
            if I.ndim == 1 :
                axs[ i, j ].plot( imgc[ ..., cnt ] )
            else :
                axs[ i, j ].imshow( imgc[ ..., cnt ], vmin = 0, vmax = .05, cmap = 'gray' )
            axs[ i, j ].set_title( 'BP' + str( i + 1  ) +'Ori' + str( j ), fontsize=12 )
            cnt += 1

        for j in range( 1 + ndir ) :
            axs[ i, j ].set_xticks( [] )
            axs[ i, j ].set_yticks( [] )
    plt.show()
