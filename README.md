# Benchmarked-Health-Claim-Predictor

### Overview
This repository contains an interactive data-visualization application built with Streamlit. The project analyzes health insurance data to uncover trends and patterns related to medical claims, patient demographics, and lifestyle factors.

Live demo: https://benchmarked-health-claim-predictor.streamlit.app

### Dataset Details
The application processes a medical insurance dataset originally containing 1,340 records and 10 columns. After cleaning, 8 duplicate rows were removed, resulting in 1,332 unique entries.

The dataset tracks the following features:
- **Patient attributes:** `Id`, `age`, `gender`, `bmi`, `bloodpressure`, `diabetic`, `children`, `smoker`, `region`
- **Financial metric:** `claim` — the medical costs billed by health insurance

### Key Data Insights
- Patient ages range from 18 to 60, with a mean age of approximately 38 years.
- Claim amounts vary significantly: minimum ≈ 1,121.87, maximum ≈ 63,770.43, mean ≈ 13,325.25.

## Technologies Used
- Python
- Streamlit
- Pandas
- Matplotlib & Seaborn

## Local Installation and Execution

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```
