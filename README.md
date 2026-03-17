# 📈 Mini-Project (ML for Time Series) - MVA 2025/2026
**Application of Independent Innovation Analysis (IIA) to Financial Data**

**Authors:** Baran Celik & Swann Cordier

### 📄 [Click here to read our full project report (PDF)](./Project_Time_Series.pdf)

---

## 💡 Project Overview
This repository is a fork of the official implementation of the paper *Independent Innovation Analysis for Nonlinear Vector Autoregressive Process* (Morioka et al., 2021). 

The goal of our project was to evaluate the ability of the IIA framework to recover meaningful latent factors from real-world financial data. Standard linear models (like the VAR model) struggle to capture nonlinear interactions and non-stationary market volatility. To address these limitations, we tested the IIA-GCL (continuous modulation) and IIA-TCL (segment-wise modulation) algorithms.

## 🛠️ Our Contributions
While we reused approximately 50% of the original authors' source code as our foundational model, we made significant contributions to adapt it to our specific use case:
* **Custom Data Pipeline:** Developed scripts for unsupervised feature extraction on high-dimensional stock market data.
* **Sensitivity Analysis:** Modified the code on simulated data to evaluate the influence of key parameters and identify the method's limitations.
* **Task Distribution:** * *Swann Cordier* focused on the implementation and analysis of the **IIA-TCL** algorithm (based on temporal segments).
  * *Baran Celik* focused on the **IIA-GCL** algorithm (based on continuous time indices).

## 📊 Datasets Used
* **Simulated Data (NVAR):** Used to validate the implementation and explore the algorithm's sensitivity.
* **Real Financial Data:** Log returns of 10 major US stocks (AAPL, MSFT, AMZN, etc.) from January 2018 to December 2023. This period includes major regime shifts, most notably the 2020 COVID-19 crash and the 2022 inflationary downturn.

## 🏆 Key Findings
Our experiments validated the ability of both frameworks to capture market non-stationarity:
* **IIA-GCL** successfully isolated systemic risk factors through continuous modulation.
* **IIA-TCL** provided a more granular decomposition, managing to separate directional shocks from symmetric volatility spikes (particularly during the COVID-19 crisis).

---

## 🚀 How to Run the Project
To run the code and reproduce our results, simply open and execute the following Jupyter Notebook:
`iia-results-notebook-celik-cordier.ipynb`
