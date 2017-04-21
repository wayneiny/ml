from common import *

################################################################################
# 3. Calculate the covariance matrix of examples

centered_faces_covariance = np.cov(centered_faces_matrix) / len(faces_list)
eigenvalues, eigenvectors = np.linalg.eig(centered_faces_covariance)

# diagonal_matrix = np.dot(np.dot(eigenvectors.T, centered_faces_covariance), eigenvectors)
# eigenvalues = np.load('eigenvalues.npy')
# eigenvectors = np.load('eigenvectors.npy')

top_k_values = eigenvalues[:400].reshape((400, 1))
top_k_basis = eigenvectors[:, :400]


################################################################################
# 4. Check result

# display the mean by using top k eigenvalues and eigenvectors
top_k_reconstruction = (np.dot(top_k_basis, top_k_values).astype(int)+faces_mean_single).astype(int).reshape((112,92))

plt.imshow(top_k_reconstruction, cmap='gray')
plt.show()

# example
coefficients = np.dot(top_k_basis.T, centered_faces_matrix[:,18])
top_k_reconstruction_example = (np.dot(top_k_basis, coefficients).reshape((10304, 1)) + faces_mean_single).astype(int).reshape((112, 92))

plt.imshow(top_k_reconstruction_example, cmap='gray')
plt.show()
