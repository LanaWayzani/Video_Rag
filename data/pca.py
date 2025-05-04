import os
import numpy as np
import joblib
from sklearn.decomposition import PCA

"""
This script loads precomputed text and combined text embeddings from the Video RAG system,
applies Principal Component Analysis (PCA) to reduce their dimensionality, and saves the 
compressed embeddings and fitted PCA model for future use ( for the vector search methods to decrease the dimensionality of the embeddings).
"""

#Defining the paths for the embeddings and the PCA model
root = "Video_Rag/data/embeddings"
text_path = os.path.join(root, "text_embeddings.npy")                    
ctp = os.path.join(root, "combined_text_embeddings.npy")              
compressed_tp = os.path.join(root, "compressed_text_embeddings.npy")    
cctp = os.path.join(root, "combined_text_embeddings_pca.npy")           
pca_path = os.path.join(root, "pca_model.pkl")                          

text_embeddings = np.load(text_path).astype("float32")
combined_text_embeddings = np.load(ctp).astype("float32")

stacked = np.vstack([text_embeddings, combined_text_embeddings])
# Fit PCA to reduce dimensions while preserving 99%+ variance
pca = PCA(n_components=544)  #directly chosen based on explained variance
pca.fit(stacked)

#Transforming the original embeddings to the new PCA space separately
compressed_text = pca.transform(text_embeddings)
compressed_combined = pca.transform(combined_text_embeddings)


np.save(compressed_tp, compressed_text)
np.save(cctp, compressed_combined)
joblib.dump(pca, pca_path)


print(f"Original dimension: {text_embeddings.shape[1]}")
print(f"Compressed dimension: {compressed_text.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
