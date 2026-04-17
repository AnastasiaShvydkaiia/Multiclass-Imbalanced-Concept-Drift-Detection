# Concept Drift Detection in Multi-class Imbalanced Data Streams
This thesis investigates concept drift detection in multi-class imbalanced data streams. ADWIN, KSWIN,  Autoencoder methods are evaluated on synthetic data streams (Random RBF, Rotating Hyperplane) and real-world Insects dataset.
## Project Structure
```bash
.
├── drift_detection.ipynb # Jupyter notebooks with synthetic stream generators and experiments
├── app.py # Streamlit demo application
├── detectors.py # AE, ADWIN and KSWIN wrapper
├── metrics.py # Metric tracker fro evaluation
├── classifier.py # Classifier wrapper
├── stream_generator.py # RandomRBF stream generator
├── data/ # Real datasets 
│ ├── INSECTS abrupt_imbalanced.csv
│ └── INSECTS gradual_imbalanced.csv
│
├── requirements.txt # Python dependencies
└── README.md 
```
## Running Streamlit demo

1. Clone the repository:
```
git clone https://github.com/AnastasiaShvydkaiia/Multiclass-Imbalanced-Concept-Drift-Detection.git
```
2. Create virtual environment:
```
python -m venv venv
source venv/bin/activate 
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run demo:
```
streamlit run app.py
```
