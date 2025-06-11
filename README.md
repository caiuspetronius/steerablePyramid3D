# steerablePyramid3D
A Steerable Pyramid Decomposition of 3D/2D/1D images along "3D Steerable Pyramid based on conic filters"
Celine A. Delle Luche, Florence Denis and Atilla Baskurt and the resulting statistical image representation based on "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" by Javier Portilla and Eero P. Simoncelli.

This is an inversible decomposition of 3D/2D/1D images into band-pass oriented components (band-pass only for 1D images). Currently, the following regularly oriented 3D conic filters are implemented: octahedral (3 directions), cubic (4 directions), icosahedral (6 directions), rhombic dodecahedral (7 directions) dodecahedral (10 directions), and rhombic triacontahedron (16 directions). Any number of spatial frequency bands can be used. Any number of orientation bands (>=3) can be used for 2D images.

A sample 3D image:
<img width="300" alt="Sample 3D image" src="https://github.com/user-attachments/assets/4bef7169-14e4-40c8-985f-c5bd18d43d09" />

Expansion of the image (central slice is shown) into 4 spatial bands and 4 orientation bands (cubic symmetry).
The absolte value (magnitude)
![Figure_3D_amp](https://github.com/user-attachments/assets/53a9912d-2c3e-4064-bcc5-64b315501105)

The real part (symmetric features)
![Figure_3D_real](https://github.com/user-attachments/assets/fb37e349-5b06-4837-8ac4-1e8bd1aafd4f)

The imaginary part (asymmetric features)
![Figure_3D_imag](https://github.com/user-attachments/assets/e73ab13e-7857-4ac6-80cc-c24062b8af83)

Expansion of the central slice (2D image) into 4 spatial bands and 4 orientation bands.
The absolte value (magnitude)
![Figure_2D_amp](https://github.com/user-attachments/assets/588310d1-4601-4b9a-9d8e-f01d7eb7e7a7)

Expansion of the central row of the central slice (1D image).
The absolte value (magnitude)
![Figure_1D_amp](https://github.com/user-attachments/assets/79f6bd5a-aac6-46ce-b6cb-72b7e2e3ba03)
