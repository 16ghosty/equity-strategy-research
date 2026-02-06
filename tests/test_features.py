"""
Unit tests for the features module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.features import (
    compute_daily_returns,
    compute_log_returns,
    compute_rolling_returns,
    compute_realized_volatility,
    compute_dollar_volume,
    compute_average_dollar_volume,
    compute_momentum_score,
    FeatureComputer,
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    np.random.seed(42)
    
    # Generate random walk prices
    returns = np.random.randn(100, 3) * 0.02  # 2% daily vol
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(
        prices,
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL']
    )


@pytest.fixture
def sample_volume():
    """Create sample volume data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    np.random.seed(42)
    
    # Random volumes between 1M and 10M
    volumes = np.random.randint(1_000_000, 10_000_000, size=(100, 3))
    
    return pd.DataFrame(
        volumes,
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL']
    )


class TestDailyReturns:
    """Tests for daily return calculation."""
    
    def test_basic_returns(self, sample_prices):
        """Test basic return calculation."""
        returns = compute_daily_returns(sample_prices)
        
        # First row should be NaN
        assert returns.iloc[0].isna().all()
        
        # Manual calculation for first valid return
        expected_aapl = sample_prices['AAPL'].iloc[1] / sample_prices['AAPL'].iloc[0] - 1
        assert np.isclose(returns['AAPL'].iloc[1], expected_aapl)
    
    def test_returns_shape(self, sample_prices):
        """Test that returns have same shape as prices."""
        returns = compute_daily_returns(sample_prices)
        assert returns.shape == sample_prices.shape
    
    def test_returns_no_lookahead(self, sample_prices):
        """Test that returns at t use only prices up to t."""
        returns = compute_daily_returns(sample_prices)
        
        # For any date t, return[t] should only depend on price[t] and price[t-1]
        for i in range(1, len(sample_prices)):
            expected = sample_prices.iloc[i] / sample_prices.iloc[i-1] - 1
            pd.testing.assert_series_equal(
                returns.iloc[i], 
                expected,
                check_names=False
            )


class TestLogReturns:
    """Tests for log return calculation."""
    
    def test_log_returns(self, sample_prices):
        """Test log return calculation."""
        log_returns = compute_log_returns(sample_prices)
        
        # Check against numpy calculation
        expected = np.log(sample_prices['AAPL'].iloc[1] / sample_prices['AAPL'].iloc[0])
        assert np.isclose(log_returns['AAPL'].iloc[1], expected)


class TestRollingReturns:
    """Tests for rolling return calculation."""
    
    def test_rolling_returns_no_skip(self, sample_prices):
        """Test rolling returns without skip period."""
        lookback = 20
        rolling = compute_rolling_returns(sample_prices, lookback=lookback, skip=0)
        
        # Check a specific calculation
        # At day 25, should be price[25] / price[5] - 1
        idx = 25
        expected = sample_prices.iloc[idx] / sample_prices.iloc[idx - lookback] - 1
        pd.testing.assert_series_equal(
            rolling.iloc[idx],
            expected,
            check_names=False
        )
    
    def test_rolling_returns_with_skip(self, sample_prices):
        """Test rolling returns with skip period (momentum)."""
        lookback = 20
        skip = 5
        rolling = compute_rolling_returns(sample_prices, lookback=lookback, skip=skip)
        
        # At day 30, should be price[30-5] / price[30-5-20] - 1 = price[25] / price[5] - 1
        idx = 30
        expected = sample_prices.iloc[idx - skip] / sample_prices.iloc[idx - skip - lookback] - 1
        pd.testing.assert_series_equal(
            rolling.iloc[idx],
            expected,
            check_names=False
        )
    
    def test_rolling_returns_nan_for_insufficient_history(self, sample_prices):
        """Test that early dates have NaN due to insufficient history."""
        lookback = 20
        skip = 5
        rolling = compute_rolling_returns(sample_prices, lookback=lookback, skip=skip)
        
        # First lookback + skip rows should be NaN
        assert rolling.iloc[:lookback + skip].isna().all().all()


class TestVolatility:
    """Tests for volatility calculation."""
    
    def test_volatility_calculation(self, sample_prices):
        """Test realized volatility calculation."""
        returns = compute_daily_returns(sample_prices)
        vol = compute_realized_volatility(returns, lookback=20)
        
        # Manual calculation for one point
        idx = 25
        manual_std = returns.iloc[idx-19:idx+1].std()
        expected_vol = manual_std * np.sqrt(252)
        
        pd.testing.assert_series_equal(
            vol.iloc[idx],
            expected_vol,
            check_names=False,
            rtol=1e-5
        )
    
    def test_volatility_positive(self, sample_prices):
        """Test that volatility is always positive."""
        returns = compute_daily_returns(sample_prices)
        vol = compute_realized_volatility(returns, lookback=20)
        
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all().all()


class TestDollarVolume:
    """Tests for dollar volume calculation."""
    
    def test_dollar_volume(self, sample_prices, sample_volume):
        """Test dollar volume calculation."""
        dv = compute_dollar_volume(sample_prices, sample_volume)
        
        # Check manual calculation
        expected = sample_prices.iloc[10] * sample_volume.iloc[10]
        pd.testing.assert_series_equal(
            dv.iloc[10],
            expected,
            check_names=False
        )
    
    def test_average_dollar_volume(self, sample_prices, sample_volume):
        """Test average dollar volume calculation."""
        adv = compute_average_dollar_volume(sample_prices, sample_volume, lookback=20)
        
        # Check that ADV is rolling mean of dollar volume
        dv = compute_dollar_volume(sample_prices, sample_volume)
        expected = dv.iloc[5:25].mean()  # Manual rolling mean at idx=24
        
        pd.testing.assert_series_equal(
            adv.iloc[24],
            expected,
            check_names=False,
            rtol=1e-10
        )


class TestMomentum:
    """Tests for momentum score calculation."""
    
    def test_momentum_with_skip(self, sample_prices):
        """Test momentum calculation with skip period."""
        mom = compute_momentum_score(sample_prices, lookback=60, skip=5)
        
        # At day 70: price[65] / price[5] - 1
        idx = 70
        expected = sample_prices.iloc[65] / sample_prices.iloc[5] - 1
        
        pd.testing.assert_series_equal(
            mom.iloc[idx],
            expected,
            check_names=False
        )
    
    def test_momentum_no_lookahead(self, sample_prices):
        """Test that momentum at t doesn't use future data."""
        lookback = 60
        skip = 5
        mom = compute_momentum_score(sample_prices, lookback=lookback, skip=skip)
        
        # For day t, momentum should only use prices up to t-skip
        for t in range(lookback + skip, len(sample_prices)):
            # Verify by manual calculation
            expected = sample_prices.iloc[t - skip] / sample_prices.iloc[t - skip - lookback] - 1
            pd.testing.assert_series_equal(
                mom.iloc[t],
                expected,
                check_names=False
            )


class TestFeatureComputer:
    """Tests for the FeatureComputer class."""
    
    def test_caching_returns(self, sample_prices):
        """Test that returns are cached."""
        fc = FeatureComputer(close_prices=sample_prices)
        
        returns1 = fc.returns
        returns2 = fc.returns
        
        # Should be the same object (cached)
        assert returns1 is returns2
    
    def test_get_volatility(self, sample_prices):
        """Test volatility getter."""
        fc = FeatureComputer(close_prices=sample_prices)
        
        vol = fc.get_volatility(lookback=20)
        assert not vol.empty
        assert vol.shape == sample_prices.shape
    
    def test_get_momentum(self, sample_prices):
        """Test momentum getter."""
        fc = FeatureComputer(close_prices=sample_prices)
        
        mom = fc.get_momentum(lookback=60, skip=5)
        assert not mom.empty
    
    def test_get_dollar_volume_requires_volume(self, sample_prices):
        """Test that dollar volume requires volume data."""
        fc = FeatureComputer(close_prices=sample_prices)
        
        with pytest.raises(ValueError, match="Volume data required"):
            fc.get_avg_dollar_volume()
    
    def test_get_dollar_volume_with_volume(self, sample_prices, sample_volume):
        """Test dollar volume with volume data."""
        fc = FeatureComputer(
            close_prices=sample_prices,
            volume=sample_volume
        )
        
        adv = fc.get_avg_dollar_volume(lookback=20)
        assert not adv.empty
