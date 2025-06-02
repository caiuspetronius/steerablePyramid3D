import os
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

'''
Based on "3D Steerable Pyramid based on conic filters" by
C´eline A. Delle Luche, Florence Denis and Atilla Baskurt
'''

def LP( R, l ) :
    # low-pass filter of order l in the interval [ 0, k ]
    a = np.pi / 4 / ( l + 1 )
    b = 2 * a
    LP = np.sqrt( 1/2 * ( 1 + np.cos( np.pi * ( R - a ) / ( b - a ) ) ) )
    LP[ R >= b ] = 0
    LP[ R <= a ] = 1
    return LP

def HP( R, l ) :
    # high-pass filter of order l in the interval [ 0, k ]
    a = np.pi / 4 / ( l + 1 )
    b = 2 * a
    HP = np.sqrt( 1/2 * ( 1 - np.cos( np.pi * ( R - a ) / ( b - a ) ) ) )
    HP[ R >= b ] = 1
    HP[ R <= a ] = 0
    return HP

def OP( X, Y, Z, R, ndir, m ) :
    # conical filter in direction m defined by a regular polyhedron with ndir directioins
    p = ( 1 + np.sqrt( 5 ) ) / 2  # golden ratio
    if ndir == 3 :  # octahedron
        V = np.array( [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ] )
    elif ndir == 4 :  # cube
        V = np.array( [ [ 1, 1, 1 ], [ 1, -1, 1 ], [ -1, 1, 1 ], [ -1, -1, 1 ] ] ) / np.sqrt( 3 )
    elif ndir == 6 :  # icosahedron
        V = np.array( [ [ p, 1, 0 ], [ p, -1, 0 ], [ 1, 0, p ], [ -1, 0, p ], [ 0, p, 1 ], [ 0, p, -1 ] ] ) / np.sqrt( p + 2 )
    elif ndir == 7 :  # rhombic dodecahedron (compound of octahedron and cube)
        VO = np.array( [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ] )
        VC = np.array( [ [ 1, 1, 1 ], [ 1, -1, 1 ], [ -1, 1, 1 ], [ -1, -1, 1 ] ] ) / np.sqrt( 3 )
        V = np.concatenate( [ VO, VC ], axis = 0 )
    elif ndir == 10 :  # dodecahedron
        V = np.array( [ [ 1, 1, 1 ], [ -1, 1, 1 ], [ 1, -1, 1 ], [ 1, 1, -1 ], [ 0, 1/p, p ], [ 0, -1/p, p ], [ 1/p, p, 0 ], [ -1/p, p, 0 ], [ p, 0, 1/p], [ p, 0, -1/p ] ] ) / np.sqrt( 3 )
    elif ndir == 16 :  # rhombic triacontahedron (compound of icosahedron and dodecahedron)
        VI = np.array( [ [ p, 1, 0 ], [ p, -1, 0 ], [ 1, 0, p ], [ -1, 0, p ], [ 0, p, 1 ], [ 0, p, -1 ] ] ) / np.sqrt( p + 2 )
        VD = np.array( [ [ 1, 1, 1 ], [ -1, 1, 1 ], [ 1, -1, 1 ], [ 1, 1, -1 ], [ 0, 1/p, p ], [ 0, -1/p, p ], [ 1/p, p, 0 ], [ -1/p, p, 0 ], [ p, 0, 1/p], [ p, 0, -1/p ] ] ) / np.sqrt( 3 )
        V = np.concatenate( [ VI, VD ], axis = 0 )
    else :
        print( 'ndir has to be one of the following: 3, 4, 6, 7, 10, 16!' )
        sys.exit()
    V = np.sqrt( 3 / ndir ) * V  # normalize the conics so that all decomposition filters add up to 1
    # # plot directions used for the decomposition
    # fig = plt.figure()
    # ax = fig.add_subplot( projection='3d' )
    # ax.scatter( [ V[ :, 0 ], -V[ :, 0 ] ], [ V[ :, 1 ], -V[ :, 1 ] ], [ V[ :, 2 ], -V[ :, 2 ] ], marker = 'o' )
    # ax.set_aspect( 'equal' )
    return ( X * V[ m, 0 ] + Y * V[ m, 1 ] + Z * V[ m,2 ] ) / R  # cosine of the angle between the polygon vertex m and the vector r

def steerablePyramid3D( dims, nsc, ndir ) :
    xd, yd, zd = dims
    X, Y, Z = np.meshgrid( np.linspace( -np.pi, np.pi, xd ), np.linspace( -np.pi, np.pi, yd ), np.linspace( -np.pi, np.pi, zd ) )    
    R = np.sqrt( X**2 + Y**2 + Z**2 )

    O = []  # orienation filters
    for d in range( ndir ) :  # loop over directions
        O.append( OP( X, Y, Z, R, ndir, d ) )

    nfilts = 2 + nsc * ndir  # the total number of filters
    A2 = np.zeros( np.append( nfilts, I.shape ) )  # all filters squared
    H = HP( R, 0 )  # zeroth-order high-pass
    L = LP( R, 0 )  # zeroth-order low-pass
    cnt = 0
    A2[ cnt, ... ] = H**2
    for sc in range( 1, nsc + 1 ) :  # loop over frequency bands
        b = HP( R, sc ) * L  # band-pass filter
        for d in range( ndir ) :  # loop over directions
            cnt += 1
            BO = O[ d ] * b  # band-pass oriented
            A2[ cnt, ... ] = BO**2
        L = LP( R, sc ) * L  # the next-scale low-pass filter
    A2[ cnt + 1, ... ] = L**2

    return A2
        
if __name__ == "__main__" :
    nsc = 4
    ndir = 4

    sample = 'sawtooth3D_img.nii.gz'
    I = nib.load( sample ).get_fdata()
    A2 = steerablePyramid3D( I.shape, nsc, ndir )

    # check that filters add up to 1
    s = np.sum( A2, axis = 0 )  # add up all filters
    print( 'Minimum filter sum value: {}'.format( np.min( s ) ) )
    print( 'Maximum filter sum value: {}'.format( np.max( s ) ) )

    # expand I into nsc frequency bands and ndir orientation bands
    F = np.fft.fftshift( np.fft.fftn( I ) )  # get the Fourier image centered with zero freq in the center
    Ic = np.zeros_like( A2 )  # filtered image components
    for i in range( A2.shape[ 0 ] ) :
        Ic[ i, ... ] = np.real( np.fft.ifftn( np.fft.ifftshift( A2[ i, ... ] * F ) ) )   # apply each filter in the Fourier space
    res = np.sum( Ic, axis = 0 )
    err = np.abs( res - I )
    print( 'Mean absolute error of the reconstruction: {}', format( np.mean( err ) ) )
    print( 'Max absolute error of the reconstruction: {}', format( np.max( err ) ) )

    # plot the expansion
    nfilts = Ic.shape[ 0 ]
    fig, axs = plt.subplots( nsc, 1 + ndir )
    fig.set_figheight( 8 )
    fig.set_figwidth( 8 * np.floor( ndir / nsc ).astype( int ) )
    sl = I.shape[ 2 ] // 2  # draw this slice of the 3D volume
    cnt = 1
    for i in range( nsc ) :
        if i == 0 :
            axs[ i, 0 ].imshow( I[ sl, :, : ], vmin = .25, vmax = .75, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Original', fontsize=12 )
        elif i == 1 :
            axs[ i, 0 ].imshow( Ic[ -1, sl, :, : ], vmin = 0, vmax = 1, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Lowest-pass', fontsize=12 )
        elif i == 2 :
            axs[ i, 0 ].imshow( Ic[ 0, sl, :, : ], vmin = 0, vmax = .2, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Highest-pass', fontsize=12 )
        elif i == 3 :
            axs[ i, 0 ].imshow( np.sum( Ic[ :, sl, :, : ], axis = 0 ), vmin = .25, vmax = .75, cmap = 'gray' )
            axs[ i, 0 ].set_title( 'Reconstruction', fontsize=12 )

        for j in range( 1, 1 + ndir ) :
            axs[ i, j ].imshow( Ic[ cnt, sl, :, : ], vmin = 0, vmax = .05, cmap = 'gray' )
            axs[ i, j ].set_title( 'BP' + str( i + 1  ) +'Ori' + str( j ), fontsize=12 )
            cnt += 1

        for j in range( 1 + ndir ) :
            axs[ i, j ].set_xticks( [] )
            axs[ i, j ].set_yticks( [] )
    plt.show()
    





 
