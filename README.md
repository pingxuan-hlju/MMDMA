# MMDMA  

## Introduction  
The project  is an implementation of a multi-perspective node feature distribution encoding and multi-scale hypergraph learning method for inferring drug-related microbes (MMDMA). 

---

## Catalogs  
- **/data**: Contains the dataset used in our method.
- **/code**: Contains the code implementation of MMDMA algorithm.
- **dataloader.py**: Processes the drug and microbial similarities, associations, embeddings, and adjacency matrices.
- **sim.py**: Calculates the drug attribute similarities based on heat kernel.
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
- **drugsimilarity.zip**: A compressed file that contains the following two files.
  - **drugfusimilarity.txt**: Includes the functional similarities among the drugs.
  - **drughesimilarity.txt**: Includes the drug attribute similarities calculated by heat kernel.
- **microbe_microbe_similarity.txt**: Contains the microbe similarities.  
- **net1.mat**: Represents the adjacency matrix of the drug-microbe heterogeneous graph.
- **Supplementary file SF2.xlsx**: Lists the top 20 candidate microbes for each drug.

---

## How to Run the Code  
1. **Data preprocessing**: Constructs the adjacency matrices, embeddings, and other inputs for training the model.  
    ```bash
    python dataloader.py
    ```  

2. **Train and test the model**.  
    ```bash
    python main.py
    ```  
