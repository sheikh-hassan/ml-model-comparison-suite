# Assignment 2: ML Algorithms Comparison

## Overview
This assignment implements a comprehensive comparison of three machine learning algorithms:
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Naïve Bayes Classifier**

Each algorithm is trained and evaluated on both:
1. **CSV Dataset** (Titanic - Binary Classification)
2. **MNIST Dataset** (Handwritten Digits - Multi-class Classification)

## Project Structure
```
assignment_2_ml_comparison/
├── train_models.py             # Training script for all algorithms
├── app.py                       # Flask application
├── requirements.txt             # Dependencies
├── models/
│   ├── csv/
│   │   ├── knn_model.pkl
│   │   ├── decision_tree_model.pkl
│   │   ├── naive_bayes_model.pkl
│   │   └── scaler.pkl
│   ├── mnist/
│   │   ├── knn_model.pkl
│   │   ├── decision_tree_model.pkl
│   │   ├── naive_bayes_model.pkl
│   │   └── scaler.pkl
│   └── comparison_stats.pkl
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── index.html
    ├── dashboard.html
    └── comparison.html
```

## Datasets

### CSV Dataset (Titanic)
- **Task**: Binary Classification (Survived/Not Survived)
- **Features**: 5 numerical features
- **Samples**: ~891 records
- **Train/Test Split**: 80/20

### MNIST Dataset
- **Task**: Multi-class Classification (10 digits: 0-9)
- **Images**: 28×28 pixel grayscale images
- **Classes**: 10
- **Train/Test Split**: 80/20

## Installation & Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models
```bash
python train_models.py
```

**Expected Output:**
```
[PHASE 1] Downloading datasets from Kaggle...
[PHASE 2] Preparing CSV Data...
[PHASE 3] Training Models on CSV Data...
[PHASE 4] Preparing MNIST Image Data...
[PHASE 5] Training Models on MNIST Data...
[PHASE 6] Saving Comparison Statistics...
```

## Running the Application

### Start Flask Server
```bash
python app.py
```

### Access the Application
- **Home Page**: http://localhost:5002
- **Dashboard**: http://localhost:5002/dashboard
- **Comparison**: http://localhost:5002/comparison
- **API Test**: http://localhost:5002/test

## Algorithms

### K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Parameters**: k=5
- **Strengths**: Non-linear patterns, flexible
- **Weaknesses**: Slow prediction, sensitive to feature scaling

### Decision Tree
- **Type**: Tree-based classification
- **Parameters**: max_depth=10
- **Strengths**: Interpretable, handles non-linear relationships
- **Weaknesses**: Prone to overfitting, feature importance bias

### Naïve Bayes
- **Type**: Probabilistic classifier
- **Parameters**: Gaussian variant
- **Strengths**: Fast training/prediction, handles high dimensions
- **Weaknesses**: Assumes feature independence, may underfit

## Evaluation Metrics

All models are evaluated on:
- **Accuracy**: Percentage of correct predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## API Endpoints

### 1. Home Page
```
GET /
```

### 2. Dashboard
```
GET /dashboard
```

### 3. Detailed Comparison
```
GET /comparison
```

### 4. Get Statistics
```
GET /api/stats
```

### 5. Get Available Algorithms
```
GET /api/algorithms
```

### 6. Predict on CSV with Specific Algorithm
```
POST /predict/csv/<algorithm>
Content-Type: application/json

{
    "Pclass": 1,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100
}
```

Replace `<algorithm>` with: `KNN`, `Decision Tree`, or `Naïve Bayes`

### 7. Predict on MNIST with Specific Algorithm
```
POST /predict/mnist/<algorithm>
Form Data:
- image: <image_file> (28x28 PNG)
```

### 8. Predict on CSV with All Algorithms
```
POST /predict/all/csv
Content-Type: application/json

{
    "Pclass": 1,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100
}
```

Returns predictions from all three algorithms for comparison.

### 9. Predict on MNIST with All Algorithms
```
POST /predict/all/mnist
Form Data:
- image: <image_file> (28x28 PNG)
```

Returns predictions from all three algorithms for comparison.

### 10. Test Connection
```
GET /test
```

## Key Features

✅ **Three Algorithms**: KNN, Decision Tree, Naïve Bayes  
✅ **Two Datasets**: CSV (binary) and MNIST (multi-class)  
✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC  
✅ **Comparative Dashboard**: Side-by-side algorithm comparison  
✅ **All-Algorithm Predictions**: Compare all algorithms on single input  
✅ **Interactive Visualizations**: Plotly charts and tables  
✅ **REST API**: Full API for programmatic access  
✅ **Model Persistence**: Save/load trained models  
✅ **Confusion Matrices**: Heatmap visualization  
✅ **Algorithm Ranking**: Best performer analysis  

## Assignment Requirements Checklist

- [x] K-Nearest Neighbors on CSV data
- [x] K-Nearest Neighbors on image data
- [x] Decision Tree on CSV data
- [x] Decision Tree on image data
- [x] Naïve Bayes on CSV data
- [x] Naïve Bayes on image data
- [x] Comparative analysis of all models
- [x] Full metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- [x] Flask application interface
- [x] Dashboard with visualizations
- [x] Confusion matrices for each algorithm
- [x] Algorithm ranking and recommendations
- [x] REST API endpoints

## Expected Performance

### CSV Dataset (Titanic)
| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------|----------|-----------|--------|----------|---------|
| KNN | ~80-82% | ~0.80 | ~0.75 | ~0.77 | ~0.85 |
| Decision Tree | ~78-81% | ~0.78 | ~0.73 | ~0.75 | ~0.82 |
| Naïve Bayes | ~75-77% | ~0.75 | ~0.68 | ~0.71 | ~0.80 |

### MNIST Dataset
| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------|----------|-----------|--------|----------|---------|
| KNN | ~95-97% | ~0.95 | ~0.95 | ~0.95 | ~0.99+ |
| Decision Tree | ~87-90% | ~0.87 | ~0.87 | ~0.87 | ~0.97 |
| Naïve Bayes | ~85-88% | ~0.85 | ~0.85 | ~0.85 | ~0.96 |

## Comparative Analysis

### Algorithm Strengths by Dataset

**CSV (Tabular) Data:**
- **KNN**: Best for this dataset, handles feature relationships well
- **Decision Tree**: Good interpretability for feature importance
- **Naïve Bayes**: Competitive with reasonable performance

**MNIST (Image) Data:**
- **KNN**: Excellent performance, captures pixel-level similarities
- **Decision Tree**: Good but slower than KNN
- **Naïve Bayes**: Reasonable but assumes feature independence (not ideal for images)

## Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning algorithms
- **Flask**: Web framework
- **NumPy & Pandas**: Data manipulation
- **PIL**: Image processing
- **Plotly**: Interactive visualizations
- **kagglehub**: Dataset download

## Troubleshooting

### Models not found
```
Error: FileNotFoundError - models/csv/knn_model.pkl not found
Solution: Run python train_models.py first
```

### Kaggle API error
```
Error: kagglehub.KaggleApiError
Solution: Ensure kagglehub is installed (pip install kagglehub)
```

### Port already in use
```
Error: Address already in use
Solution: Change port in app.py or kill existing process
```

## Performance Insights

1. **KNN excels at both datasets** - Instance-based approach captures patterns well
2. **Decision Tree provides interpretability** - Good for understanding feature importance
3. **Naïve Bayes is fastest** - Ideal for resource-constrained environments
4. **CSV data benefits from all algorithms** - Tabular features are naturally structured
5. **MNIST shows algorithm differences** - Image data benefits from sophisticated pattern matching

## Future Enhancements

- [ ] Add more algorithms (SVM, Random Forest, Ensemble methods)
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add feature importance visualization
- [ ] Support for custom datasets
- [ ] Hyperparameter tuning interface
- [ ] Export results to CSV/PDF
- [ ] Model comparison over time
- [ ] Batch prediction API

## References

- scikit-learn Documentation: https://scikit-learn.org
- KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Decision Tree: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- Naïve Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

## License
Academic Assignment - ML Course

## Contact
For issues or questions, please contact the course instructor.
# ml-model-comparison-suite
