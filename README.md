# Breast Cancer Prediction with Machine Learning

A machine learning project that predicts breast cancer diagnosis (malignant or benign) using the Wisconsin Breast Cancer Dataset and logistic regression.

##  Project Overview

This project demonstrates the complete machine learning pipeline for binary classification:
- Data acquisition from Kaggle
- Exploratory data analysis and visualization
- Data preprocessing and feature engineering
- Model training with logistic regression
- Performance evaluation with confusion matrix and accuracy metrics

##  Objective

To build a predictive model that can accurately classify breast cancer tumors as malignant (M) or benign (B) based on features computed from digitized images of breast mass.

##  Dataset

**Source**: [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) from Kaggle

**Features**: The dataset contains features computed from digitized images of breast mass, including:
- Radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points, symmetry, fractal dimension
- Mean, standard error, and "worst" values for each feature

**Target Variable**: 
- `diagnosis`: M (Malignant) or B (Benign)

##  Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `seaborn` - Statistical data visualization
  - `matplotlib` - Plotting and visualization
  - `scikit-learn` - Machine learning algorithms and tools
  - `kagglehub` - Kaggle dataset download

##  Getting Started

### Prerequisites

```bash
pip install pandas numpy seaborn matplotlib scikit-learn kagglehub
```

### Kaggle API Setup

1. Create a Kaggle account and generate API credentials
2. Set environment variables:
   ```python
   os.environ['KAGGLE_USERNAME'] = 'your_username'
   os.environ['KAGGLE_KEY'] = 'your_api_key'
   ```

### Running the Project

1. Clone this repository
2. Open `breast_cancer_prediction.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially
4. The notebook will automatically download the dataset and train the model

##  Project Structure

```
breast_cancer_prediction.ipynb    # Main Jupyter notebook
README.md                        # Project documentation
```

##  Workflow

1. **Data Acquisition**: Download dataset from Kaggle using kagglehub
2. **Data Loading**: Load data into pandas DataFrame
3. **Exploratory Data Analysis**: 
   - Check dataset shape and structure
   - Identify missing values
   - Analyze target variable distribution
4. **Data Preprocessing**:
   - Handle missing values (drop null columns)
   - Label encoding for target variable (M→1, B→0)
   - Feature scaling using StandardScaler
5. **Model Training**:
   - Split data into training (75%) and testing (25%) sets
   - Train logistic regression classifier
6. **Model Evaluation**:
   - Generate predictions on test set
   - Create confusion matrix visualization
   - Calculate accuracy score

## 📊 Results

The model achieves strong performance in classifying breast cancer tumors:

- **Model**: Logistic Regression
- **Accuracy**: [View notebook for latest results]
- **Visualization**: Confusion matrix heatmap shows detailed performance breakdown

## Key Features

- **Automated Data Download**: Uses Kaggle API for seamless dataset acquisition
- **Comprehensive EDA**: Thorough exploration of dataset characteristics
- **Feature Engineering**: Proper encoding and scaling of features
- **Model Validation**: Train-test split ensures unbiased evaluation
- **Visual Results**: Seaborn heatmap for confusion matrix visualization

## Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline
- Data preprocessing best practices
- Binary classification with logistic regression
- Model evaluation techniques
- Data visualization with seaborn

## Future Enhancements

Potential improvements for this project:
- [ ] Feature selection and dimensionality reduction
- [ ] Cross-validation for robust model evaluation
- [ ] Comparison with other algorithms (SVM, Random Forest, etc.)
- [ ] Hyperparameter tuning
- [ ] ROC curve and AUC analysis
- [ ] Feature importance analysis

##  Contributing

Feel free to fork this project and submit pull requests for improvements. All contributions are welcome!

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Wisconsin Breast Cancer Dataset creators
- Kaggle for hosting the dataset
- Scikit-learn community for excellent documentation

---

**Note**: Remember to keep your Kaggle API credentials secure and never commit them to public repositories - hence why they are placeholders. Please see the guide on Kaggle.com to create your own API token.
