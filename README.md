# MMDMA  

## Introduction  
This project implements a multi-perspective node feature distribution encoding and multi-scale hypergraph learning method for inferring drug-related microbes (MMDMA). 

---

## Catalogs  
- **/data**: Contains the dataset used in our method.
- **/code**: Contains the code implementation of the MMDMA algorithm.
- **dataloader.py**: Processes drug and microbial similarities, associations, embeddings, and adjacency matrices.
- **sim.py**: Calculates drug attribute similarities based on the heat kernel.
- **model.py**: Defines the model.
- **main.py**: Trains the model.
- **tools.py**: Contains the early stopping function.

---

## Environment  
The MMDMA code has been implemented and tested in the following development environment: 

- Python == 3.9.18 
- Matplotlib == 3.5.3
- PyTorch == 1.12.1  
- NumPy == 1.21.5
- Scikit-learn == 1.0.2

---

## Dataset  
- **drug_names.txt**: Contains the names of 1373 drugs.  
- **microbe_names.txt**: Contains the names of 173 microbes.
- **drugfusimilarity.txt**: Includes the functional similarities among the drugs.
- **drughesimilarity.txt**: Includes the drug similarities calculated based on heat kernel.
- **microbe_microbe_similarity.txt**: Contains the microbe similarities.  
- **net1.mat**: Represents the adjacency matrix of the drug-microbe heterogeneous graph.
- **Supplementary file SF2.xlsx**: Lists the top 20 candidate microbes for each drug.

---

## How to Run the Code  
1. **Data preprocessing**: Generate adjacency matrices, embeddings, and other necessary inputs for training.  
    ```bash
    python dataloader.py
    ```  

2. **Train and test the model**:  
    ```bash
    python main.py
    ```  

**Note**: Before running `main.py`, you need to create a folder, result, to store the model's training results.

