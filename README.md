# ml
Some machine learning practices

# 1-pca

Uses classical linear algebra method to perform PCA (by computing eigenvectors and eigenvalues). Also uses the concept of autoencoder implemented by tensorflow neural network to 'approximate' PCA.

Both examples use 'The ORL Database of Faces' dataset which be obtained here http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html and Thanks for the sharing.

I selected k = 400 as size of basis vectors and the reconstructions obtained by both methods are displayed below:

| Classical PCA (eigenvectors and eigenvalues) | Autoencoder PCA with neural network | Original image |
|-----|-----|-----|
| ![pca_recon_10](https://cloud.githubusercontent.com/assets/12389984/25296906/8e1d3c6e-26b8-11e7-9448-23c2c2b1884f.png) | ![pca_nn_recon_10](https://cloud.githubusercontent.com/assets/12389984/25296949/c1b04850-26b8-11e7-8ade-eeeafb8ca134.png) | ![10](https://cloud.githubusercontent.com/assets/12389984/25296824/40a5b8ee-26b8-11e7-906a-c8b52c8d5d76.png) |
| ![pca_recon_11](https://cloud.githubusercontent.com/assets/12389984/25296911/8e21379c-26b8-11e7-91b1-0dc990984c86.png) | ![pca_nn_recon_11](https://cloud.githubusercontent.com/assets/12389984/25296947/c1afcb28-26b8-11e7-91a5-0e55ca934228.png) | ![11](https://cloud.githubusercontent.com/assets/12389984/25296825/40a5fe8a-26b8-11e7-8836-dd4522638dfd.png) |
| ![pca_recon_12](https://cloud.githubusercontent.com/assets/12389984/25296909/8e208270-26b8-11e7-965a-9d5bd7792ce9.png) | ![pca_nn_recon_12](https://cloud.githubusercontent.com/assets/12389984/25296951/c1b2c620-26b8-11e7-9068-716d9ddd8066.png) | ![12](https://cloud.githubusercontent.com/assets/12389984/25296829/40abade4-26b8-11e7-8491-a1791d3d06ef.png) |
| ![pca_recon_13](https://cloud.githubusercontent.com/assets/12389984/25296908/8e1ff742-26b8-11e7-811a-28823f044def.png) | ![pca_nn_recon_13](https://cloud.githubusercontent.com/assets/12389984/25296952/c1b3a752-26b8-11e7-8d00-d365bbe0a0d3.png) | ![13](https://cloud.githubusercontent.com/assets/12389984/25296828/40a9ac42-26b8-11e7-8fe7-791529c640f6.png) |
| ![pca_recon_14](https://cloud.githubusercontent.com/assets/12389984/25296910/8e210678-26b8-11e7-8c93-54690cab515c.png) | ![pca_nn_recon_14](https://cloud.githubusercontent.com/assets/12389984/25296950/c1b13148-26b8-11e7-8ee2-135496644a2e.png) | ![14](https://cloud.githubusercontent.com/assets/12389984/25296827/40a78d22-26b8-11e7-85a0-00ff84ffa1a5.png) |
| ![pca_recon_15](https://cloud.githubusercontent.com/assets/12389984/25296912/8e23654e-26b8-11e7-9d2e-d6420c5607ac.png) | ![pca_nn_recon_15](https://cloud.githubusercontent.com/assets/12389984/25296948/c1b02442-26b8-11e7-9f75-ca20a365680c.png) | ![15](https://cloud.githubusercontent.com/assets/12389984/25296826/40a6377e-26b8-11e7-8d6e-c89b6d481367.png) |
| ![pca_recon_16](https://cloud.githubusercontent.com/assets/12389984/25296913/8e27d4a8-26b8-11e7-8634-85ddee897bdf.png) | ![pca_nn_recon_16](https://cloud.githubusercontent.com/assets/12389984/25296954/c1bd3f9c-26b8-11e7-9c21-a88d75d00425.png) | ![16](https://cloud.githubusercontent.com/assets/12389984/25296832/40b4f0d4-26b8-11e7-91de-958a0d9fa7f0.png) |
| ![pca_recon_17](https://cloud.githubusercontent.com/assets/12389984/25296915/8e31041a-26b8-11e7-9a9d-e40d88337509.png) | ![pca_nn_recon_17](https://cloud.githubusercontent.com/assets/12389984/25296955/c1be8d2a-26b8-11e7-9a4f-47f3f7c0f333.png) | ![17](https://cloud.githubusercontent.com/assets/12389984/25296831/40b2d1c8-26b8-11e7-9ec7-d929c04f2e2e.png) |
|![pca_recon_18](https://cloud.githubusercontent.com/assets/12389984/25296914/8e2c959c-26b8-11e7-9de8-3b9090b3a31c.png) | ![pca_nn_recon_18](https://cloud.githubusercontent.com/assets/12389984/25296953/c1bcedc6-26b8-11e7-9885-74972ed87f0a.png) | ![18](https://cloud.githubusercontent.com/assets/12389984/25296830/40b0ab8c-26b8-11e7-9ea1-e4434e5008d0.png) |


