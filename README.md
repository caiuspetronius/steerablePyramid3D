# steerablePyramid3D
A Steerable Pyramid Decomposition of 3D/2D/1D images along "3D Steerable Pyramid based on conic filters"
Celine A. Delle Luche, Florence Denis and Atilla Baskurt and the resulting statistical image representation based on "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients" by Javier Portilla and Eero P. Simoncelli.

This is an inversible decomposition of 3D/2D/1D images into band-pass oriented components (band-pass only for 1D images). Currently, the following regularly oriented 3D conic filters are implemented: octahedral (3 directions), cubic (4 directions), icosahedral (6 directions), rhombic dodecahedral (7 directions) dodecahedral (10 directions), and rhombic triacontahedron (16 directions). Any number of spatial frequency bands can be used. Any number of orientation bands (>=3) can be used for 2D images.

A sample 3D image:
<img width="300" alt="Sample 3D image" src="https://github.com/user-attachments/assets/4bef7169-14e4-40c8-985f-c5bd18d43d09" />

Expansion of the image (central slice is shown) into 4 spatial bands and 4 orientation bands of the cubic symmetry. 
![Figure_1](https://github.com/user-attachments/assets/fca25ed0-012f-4ede-be72-dae28a472e28)

Expansion of the central slice (2D image) into 4 spatial bands and 4 orientation bands
![Figure_2](https://github.com/user-attachments/assets/3bcd3253-f008-429b-855b-2ee6a865cf9e)

Expansion of the central row of the central slice (1D image)
![Figure_3](https://github.com/user-attachments/assets/4f67aecd-b829-4ba6-bec3-8e6c07e15fb6)
