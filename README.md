# FloodPredictionFederated
##  HELLO AFRICA ! 

**Sobek** is a team of three young engineering students from **ENSIAS** (École Nationale Supérieure d'Informatique et d'Analyse des Systèmes), united by a shared mission: empowering African communities with **equitable access to climate intelligence**.

This project, developed as part of the **AI in Africa Challenge**, tackles one of the continent’s most pressing challenges — **flood risk prediction** — through an innovative, federated, and privacy-preserving approach to machine learning.

Inspired by the principles of **data democracy**, **African data sovereignty**, and **urban resilience in low-resource settings**, the Sobek team envisions a future where **decentralized AI tools** help even the most data-scarce regions prepare for climate shocks.

We designed this repository as a modular, open-source blueprint for **multi-country collaboration**, where every city — no matter how small or underfunded — can contribute to and benefit from collective intelligence.

**Multi-Model Federated Learning for African Flood Risk Intelligence and Data Democracy**


##  Overview

FloodPredictionFederated is a suite of machine learning models developed to predict **flood year occurrence**, **flood peak magnitude**, and **flood seasonality** across African regions using a **federated learning architecture**.

This project envisions **inclusive, privacy-preserving, and data-democratic flood prediction** by training models **locally at the country level** without requiring raw data to leave the country. The predictions feed into **regional risk maps** that help with **climate adaptation, smart urban planning, and equitable water governance** across Africa.

> Our vision: **AI for data sovereignty in Africa.**

---

##  Motivation: Data Democracy for Climate Resilience

Most African nations lack centralized, high-volume data repositories for hydrology and meteorology. However, valuable datasets exist at the **local station** or **country level**. This project uses **federated learning** to enable each country to:

- Preserve data privacy and sovereignty
- Contribute to continental intelligence
- Benefit from collaboration without sacrificing autonomy

We aim to **bridge data scarcity with cooperation**, ensuring Africa can build **resilient cities and sustainable futures** powered by its own data.

---

## Federated Learning Architecture

Our approach is built on **multi-client collaboration** with either **FedAvg** or **MetaFL** aggregation strategies.

###  Models Trained

| Task                  | Model Type         | Target                                |
|-----------------------|--------------------|----------------------------------------|
| Flood Year Prediction | Binary Classifier  | Will a flood exceed the 95th percentile this year? |
| Flood Magnitude       | Regression         | What is the log of peak discharge (`Max_Q`) this year? |
| Seasonality           | Multiclass Classifier | What season is the flood peak? (Dry, PreFlood, Flood, PostFlood) |

###  Architecture Types

- **Individual Models per Task** trained federatedly.
-  **Multi-head Model** with a shared encoder and three task-specific heads. Useful in transfer/fine-tuning setups.

---

##  Data Overview

- **ADHI dataset**: Monthly discharge series from hydrological stations across Africa.
- **Station metadata**: Includes location, catchment, and land cover info.
- **Preprocessed country CSVs**: Data is split per country to simulate localized datasets.

Each country contributes independently by:

1. **Training local models** on their data
2. **Sharing only model weights**
3. **Learning from aggregated weights**


##  Project Structure

```bash
FloodPredictionFederated/
├── LSTM/                          # Local station LSTM models for time series rainfall prediction
├── GNN/                           # Graph-based modeling for catchment relationships
├── Multi-Head Federated Learning/ # Shared encoder + 3 heads (experimental)
├── fusion_predict.py              # Combines LSTM + GNN predictions
├── map_dashboard.py               # Interactive map visualizations
├── data_by_country/               # Country-level CSVs
```

---

##  Innovation

 **Federated Learning for the Global South**  
 **Multi-task architecture for flood modeling**  
 **Model fusion across modalities (GNN, LSTM)**  
 **Low-data generalization with shared representations**  
 **Aligned with the African Union's Digital Transformation Strategy**  

---

## How to Use

1. Prepare your `data_by_country.zip` with preprocessed station-level data.
2. Run federated training for each model separately:
    - `flood_year_classification.ipynb`
    - `flood_magnitude_regression.ipynb`
    - `flood_seasonality_classification.ipynb`
3. Aggregate global models using `FedAvg`.
4. Evaluate and visualize results with `map_dashboard.py`.

---

##  Future Extensions

- Add **federated meta-learning** for fast adaptation in low-data countries.
- Integrate **real-time climate APIs** for dynamic prediction.
- Expand into **drought and water quality prediction** tasks.
- Build a **community-driven African AI model registry.**

---

## Impact Goals

**Empower African governments** to keep ownership of their data  
**Enable accurate risk planning** in data-scarce regions  
**Support equitable urban development** with climate foresight  
**Promote transparency and inclusion** through open AI infrastructure  

---

##  Acknowledgments

This project was developed as part of the **AI in Africa Challenge**, led by the **Sobek** team. We extend our gratitude to the broader ecosystems that have supported our vision of democratizing flood prediction for Africa:

- **ADHI** (African Database of Hydrological Information) for foundational datasets.
- **FloodNet** and **Digital Earth Africa** for their contributions to environmental data accessibility across the continent.
- The open-source communities behind **PySyft**, **Flower**, and **FedML** — whose federated learning frameworks made our decentralized, privacy-preserving architecture possible.

This work reflects a commitment to **African-led innovation**, **community-based resilience**, and **open, inclusive AI for good**.
