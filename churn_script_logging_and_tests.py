"""
churn_script_logging_and_tests.py

Unit tests for churn_library.py functions with integrated logging.
All test results (INFO and ERROR) are written to ./logs/churn_library.log.

Run via:
    ipython churn_script_logging_and_tests.py
  or:
    python churn_script_logging_and_tests.py
  or (with pytest):
    pytest churn_script_logging_and_tests.py -v

Author: Edward Amankwah
Date: 2026-05-19
"""

import os
import logging

import pytest

# ── Logging configuration (must precede churn_library import) ─────────────────

os.makedirs('./logs', exist_ok=True)

# Force the root logger to write to file before any other module can call
# basicConfig (churn_library calls it with stream=stderr on import).
_file_handler = logging.FileHandler('./logs/churn_library.log', mode='w')
_file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_handler)

logger = logging.getLogger('churn_tests')

import churn_library as cls  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def raw_df():  # pylint: disable=redefined-outer-name
    """Load the raw dataframe once for the entire test session."""
    return cls.import_data('./data/bank_data.csv')


@pytest.fixture(scope='module')
def engineered_data(raw_df):  # pylint: disable=redefined-outer-name
    """Return the four split datasets from perform_feature_engineering."""
    return cls.perform_feature_engineering(raw_df, response='Churn')


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_import():
    """
    Test import_data:
    - The file can be found and read.
    - The resulting dataframe has rows and columns.
    - The 'Churn' target column is created.
    """
    try:
        dataframe = cls.import_data('./data/bank_data.csv')
        logging.info("test_import: SUCCESS – data loaded")
    except FileNotFoundError as err:
        logging.error("test_import: FAILED – file not found: %s", err)
        raise err

    try:
        assert dataframe.shape[0] > 0, "Dataframe has no rows"
        assert dataframe.shape[1] > 0, "Dataframe has no columns"
        assert 'Churn' in dataframe.columns, "'Churn' column not created"
        logging.info(
            "test_import: dataframe shape %s, 'Churn' column present",
            dataframe.shape,
        )
    except AssertionError as err:
        logging.error("test_import: FAILED – %s", err)
        raise err


def test_eda(raw_df):  # pylint: disable=redefined-outer-name
    """
    Test perform_eda:
    - The images/eda directory is populated with at least the expected plots.
    """
    try:
        cls.perform_eda(raw_df)
        logging.info("test_eda: perform_eda ran without exception")
    except Exception as err:  # pylint: disable=broad-except
        logging.error("test_eda: FAILED – exception during EDA: %s", err)
        raise err

    expected_files = [
        'churn_distribution.png',
        'customer_age.png',
        'marital_status.png',
        'total_trans_ct.png',
        'heatmap.png',
    ]
    try:
        for fname in expected_files:
            fpath = os.path.join('./images/eda', fname)
            assert os.path.isfile(fpath), f"Expected EDA plot not found: {fpath}"
        logging.info(
            "test_eda: SUCCESS – all %d EDA plots saved", len(expected_files)
        )
    except AssertionError as err:
        logging.error("test_eda: FAILED – %s", err)
        raise err


def test_encoder_helper(raw_df):  # pylint: disable=redefined-outer-name
    """
    Test encoder_helper:
    - New encoded columns are added to the dataframe.
    - Encoded values are numeric (float).
    - No NaN values are introduced.
    """
    category_lst = cls.CAT_COLUMNS
    try:
        df_encoded = cls.encoder_helper(
            raw_df.copy(), category_lst, response='Churn'
        )
        logging.info("test_encoder_helper: encoder_helper ran without exception")
    except Exception as err:  # pylint: disable=broad-except
        logging.error(
            "test_encoder_helper: FAILED – exception during encoding: %s", err
        )
        raise err

    try:
        for cat in category_lst:
            new_col = f'{cat}_Churn'
            assert new_col in df_encoded.columns, (
                f"Expected column '{new_col}' not found"
            )
            assert df_encoded[new_col].dtype in [float, 'float64'], (
                f"Column '{new_col}' should be float"
            )
            assert df_encoded[new_col].isna().sum() == 0, (
                f"Column '{new_col}' contains NaN values"
            )
        logging.info(
            "test_encoder_helper: SUCCESS – %d encoded columns created",
            len(category_lst),
        )
    except AssertionError as err:
        logging.error("test_encoder_helper: FAILED – %s", err)
        raise err


def test_perform_feature_engineering(raw_df):  # pylint: disable=redefined-outer-name
    """
    Test perform_feature_engineering:
    - Returns four non-empty objects.
    - Train/test split proportions are roughly 70/30.
    - Feature columns match KEEP_COLS.
    """
    try:
        x_tr, x_te, y_tr, y_te = cls.perform_feature_engineering(
            raw_df.copy(), response='Churn'
        )
        logging.info(
            "test_perform_feature_engineering: function ran without exception"
        )
    except Exception as err:  # pylint: disable=broad-except
        logging.error(
            "test_perform_feature_engineering: FAILED – %s", err
        )
        raise err

    try:
        assert x_tr.shape[0] > 0, "x_train is empty"
        assert x_te.shape[0] > 0, "x_test is empty"
        assert y_tr.shape[0] > 0, "y_train is empty"
        assert y_te.shape[0] > 0, "y_test is empty"

        total = x_tr.shape[0] + x_te.shape[0]
        test_ratio = x_te.shape[0] / total
        assert 0.25 <= test_ratio <= 0.35, (
            f"Unexpected test ratio: {test_ratio:.2f}"
        )

        for col in cls.KEEP_COLS:
            assert col in x_tr.columns, f"Missing column in x_train: {col}"

        logging.info(
            "test_perform_feature_engineering: SUCCESS – train %s, test %s",
            x_tr.shape, x_te.shape,
        )
    except AssertionError as err:
        logging.error(
            "test_perform_feature_engineering: FAILED – %s", err
        )
        raise err


def test_train_models(engineered_data):  # pylint: disable=redefined-outer-name
    """
    Test train_models:
    - Model files (.pkl) are created in ./models/.
    - Result images (ROC curve, feature importance, classification reports)
      are saved in ./images/results/.
    """
    x_tr, x_te, y_tr, y_te = engineered_data
    try:
        cls.train_models(x_tr, x_te, y_tr, y_te)
        logging.info("test_train_models: train_models ran without exception")
    except Exception as err:  # pylint: disable=broad-except
        logging.error("test_train_models: FAILED – %s", err)
        raise err

    expected_models = ['rfc_model.pkl', 'logistic_model.pkl']
    try:
        for model_file in expected_models:
            model_path = os.path.join('./models', model_file)
            assert os.path.isfile(model_path), (
                f"Expected model file not found: {model_path}"
            )
        logging.info(
            "test_train_models: SUCCESS – %d model files saved",
            len(expected_models),
        )
    except AssertionError as err:
        logging.error("test_train_models: FAILED – model files missing: %s", err)
        raise err

    expected_results = [
        'roc_curve_result.png',
        'feature_importances.png',
        'rf_results.png',
        'logistic_results.png',
    ]
    try:
        for result_file in expected_results:
            result_path = os.path.join('./images/results', result_file)
            assert os.path.isfile(result_path), (
                f"Expected result image not found: {result_path}"
            )
        logging.info(
            "test_train_models: SUCCESS – %d result images saved",
            len(expected_results),
        )
    except AssertionError as err:
        logging.error(
            "test_train_models: FAILED – result images missing: %s", err
        )
        raise err


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("===== Starting churn library tests =====")

    test_import()
    logger.info("test_import passed")

    loaded_df = cls.import_data('./data/bank_data.csv')

    test_eda(loaded_df)
    logger.info("test_eda passed")

    test_encoder_helper(loaded_df)
    logger.info("test_encoder_helper passed")

    test_perform_feature_engineering(loaded_df)
    logger.info("test_perform_feature_engineering passed")

    splits = cls.perform_feature_engineering(loaded_df.copy(), response='Churn')
    test_train_models(splits)
    logger.info("test_train_models passed")

    logger.info("===== All tests passed =====")
    print("All tests passed. See ./logs/churn_library.log for details.")
