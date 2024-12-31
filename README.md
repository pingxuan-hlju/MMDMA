# MMDMA  
**Inferring drug-related microbes through multi-perspective node feature distribution encoding and multi-scale hypergraph learning.**

---

## Operating Environment  
- `python==3.9.18`  
- `matplotlib==3.5.3`  
- `pytorch==1.12.1`  
- `numpy==1.21.5`  
- `scikit-learn==1.0.2`  

---

## File Introduction  
- **dataloader.py**: Processes drug and microbial similarities and associations, forms embeddings, adjacency matrices, etc.  
- **tools.py**: Saves the optimal parameters for the model.  
- **main.py**: Trains the model.  
- **model.py**: Defines the model.  
- **ST2.xlsx**: Lists the top 20 candidate microbes predicted for each drug.  

---

## Data  
- **drug_names**: Contains names of 1373 drugs.  
- **microbe_names**: Contains names of 173 microbes.  
- **drugsimilarity.zip**: Contains similarities between drugs.  
- **drugsimilarity.txt**: Interactions between drugs based on functional similarities.  
- **drugheatsimilarity.txt**: Interactions between drugs based on heat kernel similarities.  
- **microbe_microbe_similarity**: Contains similarities between microbes.  
- **net1.mat**: Adjacency matrix of drug and microbe heterogeneous graph.  

---

## Run Steps  
1. Run **dataloader.py**.  
2. Run **main.py**.  

**Note**: Before running **train.py**, you need to create a folder named **result** to store the model's training results.

