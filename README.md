# steerablePyramid3D
A Pytorch implementation of the Steerable Pyramid Decomposition of 3D/2D/1D images along "3D Steerable Pyramid based on conic filters"
Celine A. Delle Luche, Florence Denis and Atilla Baskurt and the resulting statistical image representation based on "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" by Javier Portilla and Eero P. Simoncelli.

This is an inversible decomposition of 3D/2D/1D images into band-pass oriented components (band-pass only for 1D images). Currently, the following regularly oriented 3D conic filters are implemented: octahedral (3 directions), cubic (4 directions), icosahedral (6 directions), rhombic dodecahedral (7 directions) dodecahedral (10 directions), and rhombic triacontahedron (16 directions). Any number of spatial frequency bands can be used. Any number of orientation bands (>=3) can be used for 2D images. Since this is in Pytorch one can run batches and pixel channels in parallel by stacking images into Batch x Channel x Height x Width x Depth Pytorch tensors.

A sample 3D image:
<img width="300" alt="Sample 3D image" src="https://github.com/user-attachments/assets/4bef7169-14e4-40c8-985f-c5bd18d43d09" />

Expansion of the image (central slice is shown) into 4 spatial bands and 4 orientation bands (cubic symmetry).

The real part (symmetric features)
![sawtooth_3D_real](https://github.com/user-attachments/assets/eb4162fa-28fd-43b4-a87e-6eced859b04c)

The imaginary part (asymmetric features)
![sawtooth_3D_imag](https://github.com/user-attachments/assets/da3d49bd-5a82-4662-8ab6-05c5d453e8a5)

Expansion of the central slice (2D image) into 4 spatial bands and 4 orientation bands.

The real part (symmetric features)
![sawtooth_2D_real](https://github.com/user-attachments/assets/baa7f02f-f876-459d-a63b-5b352408643e)

The imaginary part (asymmetric features)
![sawtooth_2D_imag](https://github.com/user-attachments/assets/9c027ec3-ef65-4689-852c-0b9dcc156b52)

Expansion of the central row of the central slice (1D image).

The real part (no imaginary part for 1D images)
![sawtooth_1D_real](https://github.com/user-attachments/assets/89596f6c-8c0b-4bb7-b588-f75d8ef89a41)


