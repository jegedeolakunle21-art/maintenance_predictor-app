# ⚙️ AI-Based Machinery Predictive Maintenance System

### Machine Learning | Industrial Equipment Monitoring | Predictive Maintenance | Mechanical Engineering

An interactive machine learning application designed to assess industrial machinery operating conditions and classify potential equipment fault categories using a trained Random Forest classifier.

The system combines equipment operating parameters, thermal conditions, pressure measurements, vibration indicators, lubrication parameters, and equipment health indicators to support predictive maintenance assessment.

Developed using Python and Streamlit, the application provides an interactive dashboard for assessing centrifugal pumps and gas turbines.

---

## 🚀 1. Project Overview

Industrial machinery such as centrifugal pumps and gas turbines operates under varying mechanical, thermal, hydraulic, and operational conditions.

Equipment deterioration can result in:
- Unexpected equipment breakdown.
- Production downtime.
- Increased maintenance expenditure.
- Reduced operational efficiency.
- Equipment damage and safety risks.

Traditional maintenance approaches often rely on scheduled servicing or corrective maintenance after equipment failure.

Predictive maintenance uses equipment condition data and analytical models to identify patterns associated with machinery faults.

This project demonstrates how machine learning can support machinery condition assessment by classifying equipment operating conditions using a trained Random Forest model.

## 🎯 2. Project Objectives

1. Develop an AI-based machinery predictive maintenance application.
2. Apply machine learning to industrial equipment fault classification.
3. Assess the relationship between operating conditions and equipment health.
4. Support condition-based maintenance decision-making.
5. Develop an interactive dashboard for machinery assessment.
6. Demonstrate the application of mechanical engineering and machine learning in industrial asset management.

## ✨ 3. Key Features

### A. Equipment Selection

The application supports two equipment categories:

- Centrifugal Pump
- Gas Turbine

### B. Fault Mode Selection

Users can select from the following fault categories:

- Cavitation
- Combustion Instability
- Compressor Degradation
- Compressor Fouling
- Impeller Degradation
- Lubrication Degradation
- Lubrication Problem
- Mechanical Seal Failure
- Normal Operating Condition
- Rotor Imbalance
- Seal Degradation
- Shaft Misalignment
- Turbine Degradation

### C. Operating Parameter Monitoring

The application accepts multiple equipment operating parameters, including:

- Operating hours.
- Equipment load.
- Rotational speed.
- Suction pressure.
- Discharge pressure.
- Suction temperature.
- Discharge temperature.
- Bearing temperature.
- Lubricating oil temperature.
- Lubricating oil pressure.
- Overall vibration.
- Axial vibration.
- Radial vibration.
- Motor current.
- Motor power.
- Flow rate.
- Equipment efficiency.
- Ambient temperature.
- Cooling water temperature.
- Cooling water flow rate.

### D. Equipment Health Indicators

The application also considers:

- Oil debris count.
- Seal leakage rate.
- Days since maintenance.
- Maintenance count.
- Degradation index.
- Health index.
- Failure-within-seven-days indicator.
- Remaining useful life input.

### E. Automatic Feature Engineering

The application calculates the pressure ratio from suction and discharge pressures:

\[
PR=\frac{P_{discharge}}{P_{suction}}
\]

It also automatically encodes equipment type and fault-mode categories into numerical model features.

### F. Machine Learning Assessment

The application generates:
- Predicted equipment classification.
- Highest prediction probability.
- Probability distribution across model classes.
- Tabular prediction results.

### G. Assessment Export

Users can download equipment assessment input data as a CSV file for further analysis and record keeping.

---

## 🧠 4. Machine Learning Architecture

**Selected algorithm: Random Forest Classifier**

The application uses a previously trained Random Forest classification model.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate classification predictions.

The model is suitable for investigating nonlinear relationships between equipment operating conditions and machinery fault classifications.

### Model workflow

```text
       EQUIPMENT SELECTION
       Pump / Gas Turbine
                |
                v
        FAULT MODE INPUT
                |
                v
      OPERATING PARAMETERS
                |
                v
       EQUIPMENT HEALTH DATA
                |
                v
        FEATURE ENGINEERING
          Pressure Ratio
                |
                v
       CATEGORICAL ENCODING
       Equipment and Faults
                |
                v
       MODEL INPUT VALIDATION
                |
                v
       RANDOM FOREST MODEL
                |
                v
       CLASSIFICATION OUTPUT
                |
          +-----+-----+
          |           |
          v           v
      PREDICTED    PREDICTION
        CLASS     PROBABILITIES
          |           |
          +-----+-----+
                |
                v
        STREAMLIT DASHBOARD
                |
                v
        ASSESSMENT EXPORT
```

## 📊 5. Model Input Parameters

The model uses equipment operating conditions and engineered features.

| Parameter | Description | Unit |
|---|---|---|
| operating_hours | Total operating hours | Hours |
| load_pct | Equipment operating load | % |
| rpm | Rotational speed | RPM |
| suction_pressure_bar | Suction pressure | bar |
| discharge_pressure_bar | Discharge pressure | bar |
| suction_temp_C | Suction temperature | °C |
| discharge_temp_C | Discharge temperature | °C |
| bearing_temp_C | Bearing temperature | °C |
| lube_oil_temp_C | Lubricating oil temperature | °C |
| lube_oil_pressure_bar | Lubricating oil pressure | bar |
| vibration_mm_s | Overall vibration | mm/s |
| axial_vibration_mm_s | Axial vibration | mm/s |
| radial_vibration_mm_s | Radial vibration | mm/s |
| motor_current_A | Motor current | A |
| motor_power_kW | Motor power | kW |
| flow_rate_m3_h | Volumetric flow rate | m³/h |
| efficiency_pct | Equipment efficiency | % |
| ambient_temp_C | Ambient temperature | °C |
| cooling_water_temp_C | Cooling water temperature | °C |
| cooling_water_flow_m3_h | Cooling water flow | m³/h |
| oil_debris_count | Oil debris indicator | Count |
| seal_leakage_rate_L_h | Seal leakage rate | L/h |
| days_since_maintenance | Time since last maintenance | Days |
| maintenance_count | Maintenance history indicator | Count |
| degradation_index | Equipment degradation indicator | Index |
| health_index | Equipment health indicator | Index |
| failure_next_7d | Seven-day failure indicator | Binary |
| rul_hours | Remaining useful life input | Hours |
| pressure_ratio | Discharge-to-suction pressure ratio | Dimensionless |

The application additionally generates one-hot encoded equipment and fault-mode features according to the feature list stored in `machinery_model_features.pkl`.

**Important:** These are application inputs. Their usefulness and validity depend on the original model training dataset, feature definitions, and preprocessing procedures.

## 📈 6. Prediction Results

After the user submits the equipment parameters, the application displays:

### Predicted Class

The predicted equipment classification returned by the Random Forest model.

### Highest Probability

The highest class probability returned by the classifier.

### Prediction Probability Distribution

The application visualizes the probabilities associated with the available model classes using a bar chart.

These probabilities represent model outputs and should not automatically be interpreted as calibrated real-world failure probabilities.

## 🏗️ 7. System Architecture

```text
         INDUSTRIAL EQUIPMENT
         +------------------+
         | Centrifugal Pump |
         | Gas Turbine      |
         +------------------+
                  |
                  v
          SENSOR / INPUT DATA
                  |
                  v
        OPERATING CONDITION DATA
                  |
                  v
         FEATURE ENGINEERING
                  |
                  v
          INPUT DATAFRAME
                  |
                  v
         TRAINED ML MODEL
                  |
                  v
        RANDOM FOREST CLASSIFIER
                  |
                  v
          FAULT CLASSIFICATION
                  |
                  v
         PREDICTION PROBABILITIES
                  |
                  v
          STREAMLIT DASHBOARD
                  |
                  v
           CSV EXPORT
```

## 🛠️ 8. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Scikit-learn | Machine learning classification |
| Pandas | Data processing |
| Joblib | Model serialization and loading |
| gdown | Google Drive model retrieval |
| GitHub | Version control |
| Streamlit Community Cloud | Application deployment |

## 📂 9. Repository Structure

```text
maintenance_predictor-app/
│
├── streamlit_app.py
│
├── machinery_model_features.pkl
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

### File descriptions

- `streamlit_app.py`: Main Streamlit application.
- `machinery_model_features.pkl`: Serialized list of expected model input features.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `LICENSE`: Project licensing information.

**Model storage:** The trained machinery prediction model is downloaded from Google Drive at application startup when it is not already available in the configured temporary directory.

## ⚙️ 10. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/jegedeolakunle21-art/maintenance_predictor-app.git
```

### Step 2: Navigate into the project directory

```bash
cd maintenance_predictor-app
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser.

## ☁️ 11. Deployment

The application is designed for deployment using Streamlit Community Cloud.

Deployment procedure:

1. Push the repository to GitHub.
2. Visit https://share.streamlit.io/
3. Sign in using your GitHub account.
4. Select the `maintenance_predictor-app` repository.
5. Select the `main` branch.
6. Set the application entry point to `streamlit_app.py`.
7. Deploy the application.

### Google Drive Model Configuration

The trained model is retrieved using Google Drive through `gdown`.

The current implementation uses a configured Google Drive file ID.

For successful deployment:

- Ensure the model file is accessible through the configured sharing permissions.
- Keep `machinery_model_features.pkl` in the GitHub repository.
- Ensure the downloaded model is complete.
- Verify that the model and feature list were generated using compatible versions of the training environment.

## 🔬 12. Engineering Applications

Potential applications of the project include:

- Centrifugal pump condition assessment.
- Gas turbine condition monitoring.
- Rotating machinery fault classification.
- Vibration-based machinery diagnostics.
- Lubrication condition assessment.
- Equipment degradation analysis.
- Maintenance planning support.
- Industrial asset management.

## 🔮 13. Future Improvements

The following developments will enhance the system's industrial relevance.

### A. Real-Time Condition Monitoring

- Integration with industrial sensors.
- Live vibration monitoring.
- Temperature and pressure monitoring.
- Motor current monitoring.
- Real-time equipment health dashboards.

### B. Predictive Maintenance Analytics

- Remaining useful life estimation.
- Failure probability calibration.
- Maintenance scheduling recommendations.
- Equipment degradation trend analysis.
- Early fault detection.

### C. Advanced Machine Learning

- Random Forest versus XGBoost comparison.
- Artificial Neural Networks.
- Support Vector Machines.
- Model explainability using SHAP.
- Cross-validation.
- Model performance comparison.

### D. Industrial Integration

- SCADA system integration.
- Industrial IoT connectivity.
- Historian database integration.
- Automated maintenance alerts.
- Equipment performance reporting.

## ⚠️ 14. Model Limitations and Safety Considerations

This application is an engineering prototype for machinery condition assessment.

Important considerations:

- Predictions depend on the quality and representativeness of the original training dataset.
- Model outputs should be validated against actual equipment inspection and maintenance records.
- Prediction probabilities are not necessarily calibrated failure probabilities.
- The selected fault mode is an input to the model, so predictions should not be interpreted as independent fault discovery.
- Inputs such as `failure_next_7d` and `rul_hours` must be checked carefully for target leakage or inappropriate use of future information.
- The model should undergo independent testing before industrial deployment.
- The application does not replace vibration analysis, equipment inspection, OEM limits, protection systems, or professional engineering judgment.

## 👨‍💻 15. Author

**Olakunle Jegede**

Mechanical Engineer | Industrial Machinery | Predictive Maintenance | Artificial Intelligence | Machine Learning | Energy Systems

### Technical Interests

- Rotating Machinery
- Mechanical Equipment Maintenance
- Thermodynamics
- Heat Transfer and Fluid Mechanics
- Computational Fluid Dynamics
- Machine Learning-Enhanced Engineering
- Industrial AI and Automation
- Energy Systems Optimization

## 📜 16. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## ⭐ Acknowledgements

This project demonstrates the integration of mechanical engineering, industrial equipment knowledge, and machine learning for intelligent machinery condition assessment.

If you find this project useful, please consider giving the repository a star.

---

**Project:** AI-Based Machinery Predictive Maintenance System

**Repository:** https://github.com/jegedeolakunle21-art/maintenance_predictor-app

**Primary Algorithm:** Random Forest Classifier

**Equipment Categories:** Centrifugal Pumps and Gas Turbines

**Application Framework:** Python and Streamlit

**Engineering Domain:** Machinery Diagnostics and Predictive Maintenance
