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

def test_stochatreat_input_format():
    """Tests that the function refuses input in the wrong format"""
    right_probs = [.1, .9]
    wrong_probs = [.1, .2]
    
    right_treats = 2
    wrong_treats = 3

    right_data = pd.DataFrame(
        data={
            "id": np.arange(100),
            "block": np.arange(100),
        }
    )
    wrong_data = pd.DataFrame(
        data={
            "id": 1,
            "block": np.arange(100),
        }
    )
    empty_data = pd.DataFrame()

    idx_col = "id"
    wrong_idx_col = 0

    wrong_size = 101

    with pytest.raises(Exception):
        stochatreat(
            data=right_data,
            block_cols=["block"],
            treats=right_treats,
            idx_col=idx_col,
            probs=wrong_probs,
            random_state=42,
        )
    
    with pytest.raises(Exception):
        stochatreat(
            data=right_data,
            block_cols=["block"],
            treats=wrong_treats,
            idx_col=idx_col,
            probs=right_probs,
            random_state=42,
        )
    
    with pytest.raises(ValueError):
        stochatreat(
            data=empty_data,
            block_cols=["block"],
            treats=right_treats,
            idx_col=idx_col,
            probs=right_probs,
            random_state=42,
        )
    
    with pytest.raises(TypeError):
        stochatreat(
            data=right_data,
            block_cols=["block"],
            treats=right_treats,
            idx_col=wrong_idx_col,
            probs=right_probs,
            random_state=42,
        )
    
    with pytest.raises(ValueError):
        stochatreat(
            data=right_data,
            block_cols=["block"],
            treats=right_treats,
            idx_col=idx_col,
            probs=right_probs,
            size=wrong_size,
            random_state=42,
        )
    
    with pytest.raises(ValueError):
        stochatreat(
            data=wrong_data,
            block_cols=["block"],
            treats=right_treats,
            idx_col=idx_col,
            probs=right_probs,
            random_state=42,
        )


def test_stochatreat_output_format():
    """Tests that the function's output is in the right format'"""
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

    assert isinstance(treatments, pd.DataFrame), "The output is not a DataFrame"
    assert "treat" in treatments.columns, "Treatment column is missing"
    assert "block_id" in treatments.columns, "Block_id column is missing"
    assert idx_col in treatments.columns, "Index column is missing"
    assert len(treatments) == size, "The size of the output does not match the sampled size"
    assert treatments['treat'].isnull().sum() == 0, "There are null assignments"