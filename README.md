# ml
Some machine learning practices

# 1-pca

Uses classical linear algebra method to perform PCA (by computing eigenvectors and eigenvalues). Also uses the concept of autoencoder implemented by tensorflow neural network to 'approximate' PCA.

Both examples use 'The ORL Database of Faces' dataset which be obtained here http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html and Thanks for the sharing.

I selected k = 400 as size of basis vectors and the reconstructions obtained by both methods are displayed below:

Using eigenvectors and eigenvalues:
![pca_recon_18](https://cloud.githubusercontent.com/assets/12389984/25266925/f51855b6-2640-11e7-9a1a-1a02a2af4bf1.png)
![pca_recon_17](https://cloud.githubusercontent.com/assets/12389984/25266929/f51f4bdc-2640-11e7-947f-cd48185a61af.png)
![pca_recon_16](https://cloud.githubusercontent.com/assets/12389984/25266926/f519baaa-2640-11e7-816e-e33f22638400.png)
![pca_recon_15](https://cloud.githubusercontent.com/assets/12389984/25266924/f51796d0-2640-11e7-819a-24781e4caf11.png)
![pca_recon_14](https://cloud.githubusercontent.com/assets/12389984/25266928/f51e0c72-2640-11e7-88ca-af6d46a4eb47.png)
![pca_recon_13](https://cloud.githubusercontent.com/assets/12389984/25266927/f51d4d6e-2640-11e7-83f2-fb41aa71e0be.png)
![pca_recon_12](https://cloud.githubusercontent.com/assets/12389984/25266931/f525cbd8-2640-11e7-870f-3f429c073dcf.png)
![pca_recon_11](https://cloud.githubusercontent.com/assets/12389984/25266930/f52321a8-2640-11e7-9335-560176f7567a.png)
![pca_recon_10](https://cloud.githubusercontent.com/assets/12389984/25266876/c7c5650e-2640-11e7-8f41-d0df00d119cf.png)
