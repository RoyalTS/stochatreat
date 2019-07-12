import pytest

import numpy as np
import pandas as pd

from stochatreat import stochatreat

np.random.seed(42)

standard_probs = [[0.1, 0.9], [1/3, 2/3], [0.5, 0.5], [2/3, 1/3], [0.9, 0.1]]


@pytest.fixture(params=[10000, 100000])
def df(request):
    N = request.param
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "dummy": [1] * N,
            "block1": np.random.randint(1, 100, size=N),
            "block2": np.random.randint(0, 2, size=N),
        }
    )

    return df


@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize(
    "block_cols", [["dummy"], ["block1"], ["block1", "block2"]]
)
def test_stochatreat_no_probs(n_treats, block_cols, df):
    """Test that overall treatment assignment proportions across all strata are as intended with equal treatment assignment probabilities"""
    treats = stochatreat(
        data=df, block_cols=block_cols, treats=n_treats, idx_col="id", random_state=42
    )

    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array([1 / n_treats] * n_treats), decimal=3
    )


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize(
    "block_cols", [["dummy"], ["block1"], ["block1", "block2"]]
)
def test_stochatreat_probs(probs, block_cols, df):
    """Test that overall treatment assignment proportions across all strata are as intended with unequal treatment assignment probabilities"""
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


@pytest.mark.parametrize("probs", [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
def test_stochatreat_no_misfits(probs):
    """Test that overall treatment assignment proportions across all strata are as intended when strata are such that there are no misfits"""
    N = 1_000_000
    blocksize = 10
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "block": np.repeat(
                np.arange(N / blocksize),
                repeats=blocksize
            )
        }
    )

    treats = stochatreat(
        data=df,
        block_cols=['block'],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


@pytest.mark.parametrize("probs", standard_probs)
def test_stochatreat_only_misfits(probs):
    """Test that overall treatment assignment proportions across all strata are as intended when strata are such that there are only misfits"""
    N = 1_000_000
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "block": np.arange(N),
        }
    )

    treats = stochatreat(
        data=df,
        block_cols=['block'],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


@pytest.fixture
def correct_params():
    params = {}
    params["probs"] = [.1, .9]
    params["treat"] = 2
    params["data"] = pd.DataFrame(
        data={
            "id": np.arange(100),
            "block": np.arange(100),
        }
    )
    params["idx_col"] = "id"
    return params


def test_stochatreat_input_format_probs(correct_params):
    """Tests that the function rejects probabilities that don't add up to one"""
    probs_not_sum_to_one = [.1, .2]
    with pytest.raises(Exception):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=probs_not_sum_to_one,
        )


def test_stochatreat_input_format_treats(correct_params):
    """Tests that the function raises an error for treatments and probs of different sizes"""
    treat_too_large = 3
    with pytest.raises(Exception):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=treat_too_large,
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_format_empty_data(correct_params):
    """Tests that the function raises an error when an empty dataframe is passed"""
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        stochatreat(
            data=empty_data,
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_format_idx_col_str(correct_params):
    """Tests that the function rejects an idx_col parameter that is not a string or None"""
    idx_col_not_str = 0
    with pytest.raises(TypeError):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=idx_col_not_str,
            probs=correct_params["probs"],
        )


def test_stochatreat_input_format_size(correct_params):
    """Tests that the function rejects a sampling size larger than the data count"""
    size_bigger_than_sampling_universe_size = 101
    with pytest.raises(ValueError):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
            size=size_bigger_than_sampling_universe_size,
        )


def test_stochatreat_input_format_idx_col_unique(correct_params):    
    """Tests that the function raises an error if the idx_col is not a primary key of the data""""
    data_with_idx_col_with_duplicates = pd.DataFrame(
        data={
            "id": 1,
            "block": np.arange(100),
        }
    )
    with pytest.raises(ValueError):
        stochatreat(
            data=data_with_idx_col_with_duplicates,
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


@pytest.fixture
def get_treatments_to_check_output():
    treats = 2
    data = pd.DataFrame(
        data={
            "id": np.arange(100),
            "block": np.arange(100),
        }
    )
    idx_col = "id"
    size = 90

    treatments = stochatreat(
        data=data,
        block_cols=["block"],
        treats=treats,
        idx_col=idx_col,
        size=size,
        random_state=42,
    )

    return treatments


def test_stochatreat_output_format_type(get_treatments_to_check_output):
    """Tests that the function's output is a pd DataFrame"""
    treatments = get_treatments_to_check_output
    assert isinstance(treatments, pd.DataFrame), "The output is not a DataFrame"
    

def test_stochatreat_output_format_treat_col(get_treatments_to_check_output):
    """Tests that the function's output contains the `treat` column"""
    treatments = get_treatments_to_check_output
    assert "treat" in treatments.columns, "Treatment column is missing"
    

def test_stochatreat_output_format_block_id_col(get_treatments_to_check_output):
    """Tests that the function's output contains the `block_id`'"""
    treatments = get_treatments_to_check_output
    assert "block_id" in treatments.columns, "Block_id column is missing"
    

def test_stochatreat_output_format_idx_col(get_treatments_to_check_output):
    """Tests that the function's output contains the `idx_col`'"""
    treatments = get_treatments_to_check_output
    assert idx_col in treatments.columns, "Index column is missing"
    

def test_stochatreat_output_format_size(get_treatments_to_check_output):
    """Tests that the function's output is of the right length'"""
    treatments = get_treatments_to_check_output
    assert len(treatments) == size, "The size of the output does not match the sampled size"
    

def test_stochatreat_output_format_nulls(get_treatments_to_check_output):
    """Tests that the function's output treatments are all non null'"""
    treatments = get_treatments_to_check_output
    assert treatments['treat'].isnull().sum() == 0, "There are null assignments"
     