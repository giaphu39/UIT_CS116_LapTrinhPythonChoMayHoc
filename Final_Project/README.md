<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS116 - Python for Machine Leanring</b></h1>



## TABLE OF CONTENTS
- [TABLE OF CONTENTS](#table-of-contents)
- [COURSE INTRODUCTION](#course-introduction)
- [INSTRUCTOR](#instructor)
- [GROUP MEMBERS](#group-members)
- [COURSE PROJECT](#course-project)
- [📝 Description \& System Overview](#-description--system-overview)
- [📁 Project Directory Structure](#-project-directory-structure)
- [🚀 Local Setup \& Running Instructions](#-local-setup--running-instructions)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. (Optional) Set up a Python virtual environment](#2-optional-set-up-a-python-virtual-environment)
  - [3. Install dependencies](#3-install-dependencies)
  - [4. Prepare the dataset](#4-prepare-the-dataset)
  - [5. Run the application](#5-run-the-application)
- [🤝 Contact \& Contributions](#-contact--contributions)

## COURSE INTRODUCTION
<a name="courseintro"></a>
* **Course Name**: Python for Machine Leanring
* **Course Code**: CS116
* **Class**: CS116.P22
* **Start Date**: 17/02/25
* **Academic Yearc**: 2025

## INSTRUCTOR
<a name="instructor"></a>

* **TS.Nguyễn Vinh Tiệp** - *tiepnv@uit.edu.vn*
* **Đàm Vũ Trọng Tài** - *taidvt@uit.edu.vn*

## GROUP MEMBERS
<a name="members"></a>
| STT    | MSSV          | Họ và Tên              | Github                                               | Email                   |
| ------ |:-------------:| ----------------------:|-----------------------------------------------------:|-------------------------:
| 1      | 23520761      | Bùi Nhật Anh Khôi      | https://github.com/KhoiBui16                         |23520761@gm.uit.edu.vn   |
| 2      | 23520662      | Nguyễn Khang Hy        | https://github.com/HyIsNoob	                         |23520662@gm.uit.edu.vn   |
| 3      | 2352004       | Đinh Lê Bình An        | https://github.com/BinhAnndapoet                     |23520004@gm.uit.edu.vn   |
| 4      | 23520713      | Vũ Gia Khang           | https://github.com/bayvai20kg                        |23520713@gm.uit.edu.vn   

## COURSE PROJECT
<a name="project"></a>
**Project Title: Hybrid Product Recommendation System**

A hybrid product recommendation system that analyzes customer behavior and shopping habits to deliver personalized product suggestions for e-commerce. The system combines content-based, collaborative filtering, and machine learning approaches to maximize personalization and recommendation accuracy.

---

## 📝 Description & System Overview
<a id="-description--system-overview"></a>

This project leverages the [Consumer Behavior and Shopping Habits Dataset](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset) (~3900 rows, 18 columns), containing key customer demographics, product interactions, ratings, and purchase histories.

**Key Features:**
- Unified Streamlit web interface: view product list (list/grid), filter, and receive personalized product recommendations.
- Hybrid recommendation engine: content-based, collaborative filtering, ensemble ML (LightGBM, CatBoost, XGBoost), and hybrid weighted scoring.
- Advanced feature engineering and preprocessing for categorical and numerical attributes.
- Model evaluation using ranking metrics (Precision@N, Recall@N, NDCG, Coverage, RMSE, etc.).
- Modular, maintainable code structure for easy extension.

**Video Demo:**  
[👉 Watch the demo video here](https://youtu.be/GDIEp9cTHi4?si=EW8wvRWOflUlzfnG)

**Website Demo:**  
[👉 website demo here](https://khoibui-recommendation-system.streamlit.app/)

---

## 📁 Project Directory Structure

<a id="-project-directory-structure"></a>

```
Final_Project/
│
├── app.py                                              
├── requirements.txt                                    
├── README.md                                           
│
├── data/                                               
│   ├── shopping_behavior_updated.csv
│   ├── shopping_behavior_processed.csv
│   ├── shopping_behavior_final_features.csv
│   └── user_item_matrix.csv
│
├── src/                                                
│   ├── __init__.py
│   ├── eda.py                                          
│   ├── preprocessing.py                                
│   ├── feature_engineering.py                          
│   ├── training.py                                     
│   ├── recommendation_models.py                        
│   └── logger.py                                       
│
├── Models/                                             
│   ├── best_model.pkl
│   ├── user_profiles.pkl
│   ├── item_features.pkl
│   └── user_item_matrix.pkl
│   └── train_df.pkl
│
├── logs/                                               
│   └── *.log
│
├── Figures/                                            
│   ├── EDA/
│   ├── Preprocessing/
│   ├── Feature_Engineering/
│   └── Training/
│
└── notebooks/                                          
    └── *.ipynb
```

---

## 🚀 Local Setup & Running Instructions

<a id="-local-setup--running-instructions"></a>

### 1. Clone the repository

```bash
git clone <https://github.com/KhoiBui16/UIT_CS116_LapTrinhPythonChoMayHoc.git>
cd UIT_CS116_LapTrinhPythonChoMayHoc/Final_Project/
```

### 2. (Optional) Set up a Python virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

- Make sure the `Data/` folder contains at least `shopping_behavior_updated.csv`.  
- Download from [Kaggle](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset) if needed.

### 5. Run the application

```bash
streamlit run app.py
```

- The app will be available at: http://localhost:8501

---

## 🤝 Contact & Contributions

<a id="-contact--contributions"></a>

- **Main Author:** [KhoiBui16] ([khoib1601@example.com](mailto:khoib1601@gmail.com))
- **Contributions, Issues, Ideas:**  
  Please open an issue or pull request on this GitHub repository, or contact via email above.


