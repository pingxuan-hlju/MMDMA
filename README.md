# MMDMA
Inferring drug-related microbes through multi-perspective node feature distribution encoding and multi-scale hypergraph learning.

# Operating environment
python=== 3.9.18
matplotlib == 3.5.3  
pytorch == 1.12.1
numpy == 1.21.5
scikit-learn == 1.0.2

# File Introduction
dataloader.py : Processing drug and microbial similarities and associations, forming embeddings, adjacency matrices, etc.  
tolls.py : In order to save better parameters for the model  
main.py: Train the model  
model.py: Define the model  
ST2.slsx: Top 20 candidate microbes predicted for each drug

# data
drug_names: contains names of 1373 drugs.
microbe_names: contains names of 173 microbes.
drugsimilarity.zip: Similarities between drugs  
drugsimilarity.txt: Interactions between drugs based on functional
drugheatsimilartiy.txt: Interactions between drugs based on heat kernel
microbe_microbe_similarity: Similarities between microbes  
net1.mat: Adjacency matrix of drug and microbe heterogeneous graph  

# run step
1. dataloader.py.
2. run main.py.

Before running train.py, you need to create one folder, result, to store the model's training result

