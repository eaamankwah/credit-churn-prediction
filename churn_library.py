"""
churn_library.py

A library of functions to identify credit card customers most likely to churn.
This module covers the full data science pipeline: EDA, feature engineering,
model training, prediction, and evaluation.

Author: Data Science Team
Date: 2026-05-16
"""

import os
import logging

import joblib
import matplotlib  # pylint: disable=wrong-import-position
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
import seaborn as sns  # pylint: disable=wrong-import-position

from sklearn.ensemble import RandomForestClassifier  # pylint: disable=wrong-import-position
from sklearn.linear_model import LogisticRegression  # pylint: disable=wrong-import-position
from sklearn.metrics import classification_report, RocCurveDisplay  # pylint: disable=wrong-import-position
from sklearn.model_selection import GridSearchCV, train_test_split  # pylint: disable=wrong-import-position
from sklearn.pipeline import Pipeline  # pylint: disable=wrong-import-position
from sklearn.preprocessing import StandardScaler  # pylint: disable=wrong-import-position


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
]

KEEP_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn',
]

EDA_PATH = './images/eda'
RESULTS_PATH = './images/results'
MODELS_PATH = './models'

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy'],
}


# ── Functions ─────────────────────────────────────────────────────────────────

def import_data(pth):
    """
    Return a dataframe for the CSV found at pth.

    input:
        pth (str): a path to the CSV file

    output:
        df (pd.DataFrame): pandas dataframe with binary 'Churn' column added
    """
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    logger.info("Data loaded successfully. Shape: %s", dataframe.shape)
    return dataframe


def perform_eda(dataframe):
    """
    Perform exploratory data analysis on dataframe and save figures to
    images/eda folder.

    Saves at least:
        - churn_distribution.png   (univariate, quantitative)
        - customer_age.png         (univariate, quantitative)
        - marital_status.png       (univariate, categorical)
        - total_trans_ct.png       (univariate, quantitative with KDE)
        - heatmap.png              (bivariate / correlation)

    input:
        dataframe (pd.DataFrame): pandas dataframe

    output:
        None
    """
    os.makedirs(EDA_PATH, exist_ok=True)

    # Univariate – quantitative: Churn distribution
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PATH, 'churn_distribution.png'))
    plt.close()

    # Univariate – quantitative: Customer Age
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PATH, 'customer_age.png'))
    plt.close()

    # Univariate – categorical: Marital Status
    plt.figure(figsize=(20, 10))
    dataframe['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Marital Status Distribution (Normalized)')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PATH, 'marital_status.png'))
    plt.close()

    # Univariate – quantitative with KDE: Total Transaction Count
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Transaction Count Distribution')
    plt.xlabel('Total Trans Ct')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PATH, 'total_trans_ct.png'))
    plt.close()

    # Bivariate: Correlation heatmap
    plt.figure(figsize=(20, 10))
    numeric_df = dataframe.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PATH, 'heatmap.png'))
    plt.close()

    logger.info("EDA complete. Plots saved to %s", EDA_PATH)


def encoder_helper(dataframe, category_lst, response):
    """
    Turn each categorical column into a new column containing the mean churn
    rate for that category (mean encoding). Refactored from cell 15 of the
    original notebook to use an efficient loop over category_lst.

    input:
        dataframe (pd.DataFrame): pandas dataframe
        category_lst (list): list of column names that contain categorical features
        response (str): name of the response/target column (e.g. 'Churn')

    output:
        dataframe (pd.DataFrame): dataframe with new encoded columns appended
    """
    for category in category_lst:
        group_means = dataframe.groupby(category)[response].mean()
        dataframe[f'{category}_Churn'] = dataframe[category].map(group_means)
    logger.info("Encoding complete for columns: %s", category_lst)
    return dataframe


def perform_feature_engineering(dataframe, response):
    """
    Encode categorical features and split into train / test sets.

    input:
        dataframe (pd.DataFrame): pandas dataframe
        response (str): name of the response/target column (e.g. 'Churn')

    output:
        x_train (pd.DataFrame): X training data
        x_test  (pd.DataFrame): X testing data
        y_train (pd.Series):   y training data
        y_test  (pd.Series):   y testing data
    """
    dataframe = encoder_helper(dataframe, CAT_COLUMNS, response)

    y_target = dataframe[response]
    x_features = dataframe[KEEP_COLS].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, random_state=42
    )
    logger.info(
        "Feature engineering complete. Train size: %s, Test size: %s",
        x_train.shape, x_test.shape,
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Produce classification reports for training and testing results and store
    each report as a PNG image in images/results/.

    input:
        y_train (pd.Series):         training response values
        y_test  (pd.Series):         test response values
        y_train_preds_lr (np.array): training predictions from logistic regression
        y_train_preds_rf (np.array): training predictions from random forest
        y_test_preds_lr  (np.array): test predictions from logistic regression
        y_test_preds_rf  (np.array): test predictions from random forest

    output:
        None
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)

    reports = {
        'rf_results.png': [
            ('Random Forest Train', y_train, y_train_preds_rf),
            ('Random Forest Test', y_test, y_test_preds_rf),
        ],
        'logistic_results.png': [
            ('Logistic Regression Train', y_train, y_train_preds_lr),
            ('Logistic Regression Test', y_test, y_test_preds_lr),
        ],
    }

    for filename, pairs in reports.items():
        plt.rc('figure', figsize=(6, 6))
        _, axes = plt.subplots()
        axes.axis('off')
        y_pos = 0.95
        for title, y_true, y_pred in pairs:
            report_text = classification_report(y_true, y_pred)
            axes.text(0.01, y_pos, title, fontsize=10,
                      fontproperties='monospace', transform=axes.transAxes,
                      verticalalignment='top')
            y_pos -= 0.07
            axes.text(0.01, y_pos, report_text, fontsize=8,
                      fontproperties='monospace', transform=axes.transAxes,
                      verticalalignment='top')
            y_pos -= 0.45

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, filename), bbox_inches='tight')
        plt.close()

    logger.info("Classification report images saved to %s", RESULTS_PATH)


def feature_importance_plot(model, x_data, output_pth):
    """
    Create and store a feature importance bar chart.

    input:
        model     : fitted model object that exposes feature_importances_
        x_data    (pd.DataFrame): dataframe of X values (column names used for labels)
        output_pth (str): full file path (including filename) for the saved figure

    output:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()
    logger.info("Feature importance plot saved to %s", output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    Train Random Forest (via GridSearchCV) and Logistic Regression models,
    store result images (ROC curve, classification reports, feature importance),
    and persist the models as .pkl files under ./models/.

    input:
        x_train (pd.DataFrame): X training data
        x_test  (pd.DataFrame): X testing data
        y_train (pd.Series):   y training data
        y_test  (pd.Series):   y testing data

    output:
        None
    """
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Random Forest with hyperparameter search
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(x_train, y_train)
    best_rfc = cv_rfc.best_estimator_

    # Logistic Regression with feature scaling
    lrc = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=3000)),
    ])
    lrc.fit(x_train, y_train)

    # Predictions
    y_train_preds_rf = best_rfc.predict(x_train)
    y_test_preds_rf = best_rfc.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # ROC Curve
    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    RocCurveDisplay.from_estimator(
        best_rfc, x_test, y_test, ax=axes, alpha=0.8, name='Random Forest'
    )
    RocCurveDisplay.from_estimator(
        lrc, x_test, y_test, ax=axes, alpha=0.8, name='Logistic Regression'
    )
    plt.title('ROC Curves')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'roc_curve_result.png'))
    plt.close()
    logger.info("ROC curve saved.")

    # Classification report images
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf,
    )

    # Feature importance
    feature_importance_plot(
        best_rfc,
        pd.concat([x_train, x_test]),
        os.path.join(RESULTS_PATH, 'feature_importances.png'),
    )

    # Persist models
    joblib.dump(best_rfc, os.path.join(MODELS_PATH, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(MODELS_PATH, 'logistic_model.pkl'))
    logger.info("Models saved to %s", MODELS_PATH)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = './data/bank_data.csv'

    raw_df = import_data(DATA_PATH)
    perform_eda(raw_df)

    x_tr, x_te, y_tr, y_te = perform_feature_engineering(
        raw_df, response='Churn'
    )
    train_models(x_tr, x_te, y_tr, y_te)
