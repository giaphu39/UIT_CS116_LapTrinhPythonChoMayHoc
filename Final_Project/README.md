# 📊 CS116 Product Recommendation System

---

## 📑 Table of Contents

* [📊 CS116 Product Recommendation System](#-cs116-product-recommendation-system)

  * [📝 Project Description](#-project-description)
  * [📁 Project Structure](#-project-structure)
  * [🚀 Features](#-features)
  * [⚙️ How to Run Locally](#-how-to-run-locally)
  * [🛠 Tech Stack](#-tech-stack)
  * [📄 License](#-license)

---

## 📝 Project Description

This project is a final assignment for the course **CS116 - Lập trình Python cho Máy học (UIT)**. It implements a **hybrid recommendation system** that analyzes customer shopping behavior and habits to provide **personalized product recommendations**.

It combines collaborative filtering and attribute-based filtering, using both customer behavior and product features to enhance recommendation accuracy.

---

## 📁 Project Structure

```
Final_Project/
├── app.py                                # Streamlit entry point
├── requirements.txt                      # Python dependencies
├── Src/
│   ├── CS116_Product_Recommendation_System.ipynb   # Jupyter notebook for exploration & demo
│   └── cs116_product_recommendation_system.py     # Optional script version of the system
├── models/
│   ├── recommendation_model.py                   # Main hybrid recommendation logic
│   └── attribute_based_recommendation.py         # Attribute-based filtering module
├── data/
│   ├── shopping_behavior_updated.csv
│   ├── shopping_behavior_processed.csv
│   ├── shopping_behavior_final_features.csv
│   ├── shopping_behavior_final_features_with_customer_id.csv
│   ├── shopping_trends.csv
│   └── user_item_matrix.csv                      # Processed matrices & CSVs used by the model
```

---

## 🚀 Features

* 📌 **Hybrid Recommendation**: Combines collaborative filtering & attribute-based methods.
* 📌 **Streamlit Interface**: Interactive app for generating recommendations based on user behavior.
* 📌 **Modular Codebase**: Easy to understand, extend, and integrate with other projects.
* 📌 **Jupyter Notebook Included**: For testing, visualization, and experimentation.

---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/KhoiBui16/UIT_CS116_LapTrinhPythonChoMayHoc.git
cd UIT_CS116_LapTrinhPythonChoMayHoc/Final_Project
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 5. Open the Application

Go to your browser and visit:

```
http://localhost:8501/
```

---

## 🛠 Tech Stack

* Python 3.x
* Streamlit >= 1.32.0
* pandas >= 2.1.0
* numpy >= 1.24.0
* scikit-learn >= 1.2.0
* streamlit-extras >= 0.3.5

---

## 📄 License

This project is part of an academic submission and is licensed under the **MIT License**.
See the [LICENSE](../LICENSE) file for details.
