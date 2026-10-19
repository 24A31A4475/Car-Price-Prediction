# 🚗 Car Price Prediction

An interactive machine learning application built with Streamlit and Scikit-Learn that estimates the resale selling price of used vehicles based on specifications, usage metrics, and powertrain configuration.

---

## 📌 Features

* **Interactive Valuation Interface:** Input car details (Year, Showroom Price, Kilometers Driven, Fuel Type, Transmission, Selling Channel, and Ownership history) in a clean, multi-column dashboard layout.
* **Instant Market Estimates:** Leverages a pre-trained **Random Forest Regressor** model to compute real-time resale price predictions in Indian Rupees (Lakhs).
* **Depreciation Analytics:** Automatically computes and displays the estimated percentage of vehicle depreciation relative to its original showroom price.
* **Automated Feature Engineering:** Calculates the vehicle's age dynamically based on its manufacturing year.

---

## 🛠️ Tech Stack & Dependencies

* **Python** (3.10+)
* **Streamlit** (Interactive Frontend UI)
* **Pandas** (Data Manipulation & Preprocessing)
* **NumPy** (Numerical Processing)
* **Scikit-Learn** (Model Inference Pipeline)
* **Pickle** (Model Serialization)

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```Bash
git clone [https://github.com/24A31A4475/Car-Price-Prediction.git](https://github.com/24A31A4475/Car-Price-Prediction.git)
cd Car-Price-Prediction
```
### 2. Create and Activate a Virtual Environment
```Bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Requirements
```Bash
pip install -r requirements.txt
```
### 4. Launch the Streamlit App
```Bash
streamlit run app.py
```
