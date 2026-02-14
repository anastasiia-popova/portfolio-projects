# Data Science Portfolio

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


This collection showcases my recent data-driven projects. It is presented in the format of Jupyter Notebooks, presentations, and reports.

## Content

### Post-Training Compression for Dermatology Classification

The project studies post-training compression of a MedMNIST image classifier (DINOv3 ConvNeXt-Tiny backbone) to achieve a smaller model size and faster CPU inference with minimal loss in predictive performance. It compares INT8 quantization, structured and unstructured magnitude pruning, and combined P→FT→Q pipelines, reporting ACC/AUC/F1, serialized size, and CPU latency.

- **Status:** ✅ *Done*  |  [[GitHub repo link]](https://github.com/anastasiia-popova/compressed-medvision-classifier)
- **Tools:** PyTorch · DINOv3 · MedMNIST 
- **Methods:** Post-training Quantization (INT8/FBGEMM) · Magnitude Pruning · Classification 

### Deep Reinforcement Learning from Visual Inputs: Quantifying Representational Capacity and Architectural Inductive Bias

The project aims to study how encoder architecture choices (MLP vs. CNN) in deep reinforcement learning from images impact the optimization dynamics, sample efficiency, training stability, and generalization behavior of policy-gradient methods.

- **Status:** ✅ *Done*  | [[GitHub repo link]](https://github.com/anastasiia-popova/visual-rl-encoders)
- **Tools:** PyTorch · ALE · Wandb
- **Methods:** Reinforcement Learning · A2C · CNNs vs MLPs 

### Comparative Study of Adaptive Algorithms  

This work compares **adaptive optimization algorithms** (AdaGrad, AdaGrad-Norm, RMSprop, Adam) with gradient descent, stochastic gradient descent, and Quasi-Newton methods through both **theoretical analysis** (convergence rates, complexity, memory) and **empirical evaluation** on binary and multiclass classification tasks with convex and non-convex losses.  

- **Status:** ✅ *Done* 
- **Tools:** PyTorch · scikit-learn 
- **Methods:** Adaptive Optimizers (AdaGrad, AdaGrad-Norm, RMSprop, Adam) · Convergence & Complexity Analysis

### Differentially Private Learning for Blood Cell Image Classification  

This project explores **differentially private learning** for blood cell image classification using the **BloodMNIST dataset**.  
It focuses on training a **compact CNN with DP-SGD**, comparing its performance to a non-private model to assess the **privacy–utility trade-off**.  
 
- **Status:** ✅ *Done*  
- **Tools:** PyTorch · Opacus · MedMNIST  
- **Methods:** CNNs · DP-SGD · Hyperparameter Tuning  


### Analysis of CITE-seq data from a brain organoid
  
This project processes and analyzes **CITE-seq single-cell data** from a brain organoid,  
covering the full pipeline from **raw data processing** to **dimensionality reduction** and visualization.  

- **Status:** ✅ *Done*  
- **Tools:** alevin-fry · salmon · scanpy · matplotlib  
- **Methods:** Raw Data Processing · EDA · Data Cleaning · Dimensionality Reduction  

### Building RAG-Powered Chatbot

This project develops a **Retrieval Augmented Generation (RAG)** chatbot, implementing data  
cleaning, retrieval, and reranking to provide context-aware responses for university documents.  

- **Status:** ✅ *Done*  
- **Tools:** MongoDB · LLM APIs · BeautifulSoup  
- **Methods:** Retrieval Augmented Generation · Web Scraping · Data Cleaning · Reranking  
- **Implementation:** [GitHub – RAG Workshop](https://github.com/datomo/rag-workshop/tree/main/builder/partnerproduct)  


### In Silico Prediction of Tartrazine Toxicity

This project leverages **computational toxicology tools** to model and analyse the potential toxicity  
of **tartrazine**, a widely used synthetic dye.  

- **Status:** ✅ *Done*  
- **Tools:** VirtualToxLab · PanScreen · Molinspiration · ProTox-3.0 · SwissADME  
- **Methods:** Computer Modeling of Small Molecule Toxicity  

### Heart Attack Prediction Project
  
- **Status:** ✅ *Done*  
- **Tools:** pandas · scikit-learn · seaborn · matplotlib  
- **Methods:** EDA · Logistic Regression · SVM · XGBoost  

---
If you found these projects interesting and would like to discuss the portfolio further or explore potential collaboration possibilities, please don't hesitate to reach out to me on [LinkedIn][linkedin-url].

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
[license-url]: https://opensource.org/license/mit/
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/nastya-popova/
[red-color]: #f03c15
[orange-color]:#f07815
[green-color]: #a9c746
[blue-color]: #1589F0
