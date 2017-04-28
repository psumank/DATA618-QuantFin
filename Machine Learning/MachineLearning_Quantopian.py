from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import Latest, CustomFactor, SimpleMovingAverage, AverageDollarVolume, Returns, RSI
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.filters import Q500US, Q1500US
from quantopian.pipeline.data.quandl import fred_usdontd156n as libor


import quantopian.experimental.optimize as opt

import talib
import pandas as pd
import numpy as np
from time import time
from collections import OrderedDict

from scipy import stats
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics, svm

####################################

#Primary Source: https://www.quantopian.com/posts/machine-learning-on-quantopian

#Uses Machine Learning on a Factor-based workflow.
#Retrains the model periodically.
#Trades a large universe of stocks, using the Q1500US universe (choose a subset of stocks here).
#Beta-neutral by going long-short.
#Sector-neutral due to new optimization API.
#Sets strict limits on maximum weight of any individual stock.

# Global configuration of strategy

N_STOCKS_TO_TRADE = 100 # Will be split 50% long and 50% short
ML_TRAINING_WINDOW = 21 # Number of days to train the classifier on, easy to run out of memory here
PRED_N_FWD_DAYS = 1 # train on returns over N days into the future
TRADE_FREQ = date_rules.week_start() # How often to trade, for daily, set to date_rules.every_day()

#################################
# Definition of alphas

# Pipeline factors
bs = morningstar.balance_sheet
cfs = morningstar.cash_flow_statement
is_ = morningstar.income_statement
or_ = morningstar.operation_ratios
er = morningstar.earnings_report
v = morningstar.valuation
vr = morningstar.valuation_ratios

class Sector(Sector):
    window_safe = True
    
def make_factors():
    def Asset_Growth_3M():
        return Returns(inputs=[bs.total_assets], window_length=63)

    def Asset_To_Equity_Ratio():
        return bs.total_assets.latest / bs.common_stock_equity.latest

    def Capex_To_Cashflows():
        return (cfs.capital_expenditure.latest * 4.) / \
            (cfs.free_cash_flow.latest * 4.)
        
    def EBITDA_Yield():
        return (is_.ebitda.latest * 4.) / \
            USEquityPricing.close.latest        

    def EBIT_To_Assets():
        return (is_.ebit.latest * 4.) / \
            bs.total_assets.latest
               
    def Return_On_Total_Invest_Capital():
        return or_.roic.latest
    
    class Mean_Reversion_1M(CustomFactor):
        inputs = [Returns(window_length=21)]
        window_length = 252

        def compute(self, today, assets, out, monthly_rets):
            out[:] = (monthly_rets[-1] - np.nanmean(monthly_rets, axis=0)) / \
                np.nanstd(monthly_rets, axis=0)
                
    class MACD_Signal_10d(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 60

        def compute(self, today, assets, out, close):

            sig_lines = []

            for col in close.T:
                # get signal line only
                try:
                    _, signal_line, _ = talib.MACD(col, fastperiod=12,
                                                   slowperiod=26, signalperiod=10)
                    sig_lines.append(signal_line[-1])
                # if error calculating, return NaN
                except:
                    sig_lines.append(np.nan)
            out[:] = sig_lines 
            
    class Moneyflow_Volume_5d(CustomFactor):
        inputs = [USEquityPricing.close, USEquityPricing.volume]
        window_length = 5

        def compute(self, today, assets, out, close, volume):
            mfvs = []
            for col_c, col_v in zip(close.T, volume.T):
                # denominator
                denominator = np.dot(col_c, col_v)
                # numerator
                numerator = 0.
                for n, price in enumerate(col_c.tolist()):
                    if price > col_c[n - 1]:
                        numerator += price * col_v[n]
                    else:
                        numerator -= price * col_v[n]
                mfvs.append(numerator / denominator)
            out[:] = mfvs  
           
    def Net_Income_Margin():
        return or_.net_margin.latest           

    def Operating_Cashflows_To_Assets():
        return (cfs.operating_cash_flow.latest * 4.) / \
            bs.total_assets.latest

    def Price_Momentum_3M():
        return Returns(window_length=63)
    
    class Price_Oscillator(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            four_week_period = close[-20:]
            out[:] = (np.nanmean(four_week_period, axis=0) /
                      np.nanmean(close, axis=0)) - 1.
    
    def Returns_39W():
        return Returns(window_length=215)
    
    class Trendline(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252

        # using MLE for speed
        def compute(self, today, assets, out, close):

            # prepare X matrix (x_is - x_bar)
            X = range(self.window_length)
            X_bar = np.nanmean(X)
            X_vector = X - X_bar
            X_matrix = np.tile(X_vector, (len(close.T), 1)).T

            # prepare Y matrix (y_is - y_bar)
            Y_bar = np.nanmean(close, axis=0)
            Y_bars = np.tile(Y_bar, (self.window_length, 1))
            Y_matrix = close - Y_bars

            # prepare variance of X
            X_var = np.nanvar(X)

            # multiply X matrix an Y matrix and sum (dot product)
            # then divide by variance of X
            # this gives the MLE of Beta
            out[:] = (np.sum((X_matrix * Y_matrix), axis=0) / X_var) / \
                (self.window_length)
        
    class Vol_3M(CustomFactor):
        inputs = [Returns(window_length=2)]
        window_length = 63

        def compute(self, today, assets, out, rets):
            out[:] = np.nanstd(rets, axis=0)
            
    def Working_Capital_To_Assets():
        return bs.working_capital.latest / bs.total_assets.latest
    
    class AdvancedMomenutm(CustomFactor):
        """ Momentum factor """
        inputs = [USEquityPricing.close,
                  Returns(window_length=126)]
        window_length = 252

        def compute(self, today, assets, out, prices, returns):
            out[:] = ((prices[-21] - prices[-252])/prices[-252] -
                      (prices[-1] - prices[-21])/prices[-21]) / np.nanstd(returns, axis=0)    
    

    # Commenting out some factors to not run out-of-memory
    all_factors = {
        'Asset Growth 3M': Asset_Growth_3M,
        #'Asset to Equity Ratio': Asset_To_Equity_Ratio,
        #'Capex to Cashflows': Capex_To_Cashflows,
        #'EBIT to Assets': EBIT_To_Assets,
        #'EBITDA Yield': EBITDA_Yield,        
        'MACD Signal Line': MACD_Signal_10d,
        'Mean Reversion 1M': Mean_Reversion_1M,
        #'Moneyflow Volume 5D': Moneyflow_Volume_5d,
        'Net Income Margin': Net_Income_Margin,        
        #'Operating Cashflows to Assets': Operating_Cashflows_To_Assets,
        'Price Momentum 3M': Price_Momentum_3M,
        'Price Oscillator': Price_Oscillator,
        'Return on Invest Capital': Return_On_Total_Invest_Capital,
        '39 Week Returns': Returns_39W,
        'Trendline': Trendline,
        'Vol 3M': Vol_3M,
        'Advanced Momentum': AdvancedMomenutm,
    }        
    
    return all_factors

def shift_mask_data(X, Y, upper_percentile=70, lower_percentile=30, n_fwd_days=1):
    # Shift X to match factors at t to returns at t+n_fwd_days (we want to predict future returns after all)
    shifted_X = np.roll(X, n_fwd_days, axis=0)
    
    # Slice off rolled elements
    X = shifted_X[n_fwd_days:]
    Y = Y[n_fwd_days:]
    
    n_time, n_stocks, n_factors = X.shape
    
    # Look for biggest up and down movers
    upper = np.nanpercentile(Y, upper_percentile, axis=1)[:, np.newaxis]
    lower = np.nanpercentile(Y, lower_percentile, axis=1)[:, np.newaxis]
  
    upper_mask = (Y >= upper)
    lower_mask = (Y <= lower)
    
    mask = upper_mask | lower_mask # This also drops nans
    mask = mask.flatten()
    
    # Only try to predict whether a stock moved up/down relative to other stocks
    Y_binary = np.zeros(n_time * n_stocks)
    Y_binary[upper_mask.flatten()] = 1
    Y_binary[lower_mask.flatten()] = -1
    
    # Flatten X
    X = X.reshape((n_time * n_stocks, n_factors))

    # Drop stocks that did not move much (i.e. are in the 30th to 70th percentile)
    X = X[mask]
    Y_binary = Y_binary[mask]
    
    return X, Y_binary

def get_last_values(input_data):
    last_values = []
    for dataset in input_data:
        last_values.append(dataset[-1])
    return np.vstack(last_values).T

# Definition of Machine Learning factor which trains a model and predicts forward returns
class ML(CustomFactor):
    init = False

    def compute(self, today, assets, out, returns, *inputs):
        # inputs is a list of factors, for example, assume we have 2 alpha signals, 3 stocks,
        # and a lookback of 2 days. Each element in the inputs list will be data of
        # one signal, so len(inputs) == 2. Then each element will contain a 2-D array
        # of shape [time x stocks]. For example:
        # inputs[0]:
        # [[1, 3, 2], # factor 1 rankings of day t-1 for 3 stocks  
        #  [3, 2, 1]] # factor 1 rankings of day t for 3 stocks
        # inputs[1]:
        # [[2, 3, 1], # factor 2 rankings of day t-1 for 3 stocks
        #  [1, 2, 3]] # factor 2 rankings of day t for 3 stocks

        if (not self.init) or (today.weekday() == 0): # Monday
            # Instantiate sklearn objects
            self.imputer = preprocessing.Imputer()
            self.scaler = preprocessing.MinMaxScaler()
            log.debug('Training classifier...')
            self.clf = ensemble.AdaBoostClassifier(n_estimators=100)
            #self.clf = ensemble.RandomForestClassifier()
            
            # Stack factor rankings
            X = np.dstack(inputs) # (time, stocks, factors)
            Y = returns # (time, stocks)
        
            # Shift data to match with future returns and binarize 
            # returns based on their 
            X, Y = shift_mask_data(X, Y, n_fwd_days=PRED_N_FWD_DAYS)
            
            X = self.imputer.fit_transform(X)            
            X = self.scaler.fit_transform(X)
            
            # Fit the classifier
            self.clf.fit(X, Y)
            #log.debug(self.clf.feature_importances_)
            self.init = True

        # Predict
        # Get most recent factor values (inputs always has the full history)
        last_factor_values = get_last_values(inputs)
        last_factor_values = self.imputer.transform(last_factor_values)
        last_factor_values = self.scaler.transform(last_factor_values)

        # Predict the probability for each stock going up 
        # (column 2 of the output of .predict_proba()) and
        # return it via assignment to out.
        out[:] = self.clf.predict_proba(last_factor_values)[:, 1]
         
def make_ml_pipeline(factors, universe, window_length=21, n_fwd_days=5):
    factors_pipe = OrderedDict()
    # Create returns over last n days.
    factors_pipe['Returns'] = Returns(inputs=[USEquityPricing.open],
                                      mask=universe, window_length=n_fwd_days + 1)
    
    # Instantiate ranked factors
    for name, f in factors.iteritems():
        factors_pipe[name] = f().rank(mask=universe)
        
    # Create our ML pipeline factor. The window_length will control how much
    # lookback the passed in data will have.
    factors_pipe['ML'] = ML(inputs=factors_pipe.values(), 
                            window_length=window_length + 1, 
                            mask=universe)
 
    factors_pipe['Sector'] = Sector()
    
    pipe = Pipeline(screen=universe, columns=factors_pipe)
    
    return pipe

###########################################################
## Algo definition
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    
    schedule_function(my_rebalance, TRADE_FREQ,
                      time_rules.market_open(minutes=10))
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(),
                      time_rules.market_close())

    # Set up universe, alphas and ML pipline
    context.universe = Q1500US() # & ~IsAnnouncedAcqTarget()
    ml_factors = make_factors()
    ml_pipeline = make_ml_pipeline(ml_factors, 
                                   context.universe, n_fwd_days=PRED_N_FWD_DAYS,
                                   window_length=ML_TRAINING_WINDOW)
    # Create our dynamic stock selector.
    attach_pipeline(ml_pipeline, 'alpha_model')
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.predicted_probs = pipeline_output('alpha_model')['ML']
    context.predicted_probs.index.rename(['date', 'equity'], inplace=True)
    
    context.risk_factors = pipeline_output('alpha_model')[['Vol 3M', 'Sector']]
    context.risk_factors.index.rename(['date', 'equity'], inplace=True)
    context.risk_factors.Sector = context.risk_factors.Sector.map(Sector.SECTOR_NAMES)
    
    # These are the securities that we are interested in trading each day.
    context.security_list = context.predicted_probs.index

    
########################################################
# Portfolio construction

def my_rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    
    risk_model_factors = context.risk_factors
    risk_model_factors = risk_model_factors.join(context.predicted_probs, how='right').dropna()
    
    predictions = risk_model_factors.ML
    
    # Filter out stocks that can not be traded
    predictions = predictions.loc[data.can_trade(predictions.index)]
    # Select top and bottom N stocks
    predictions = pd.concat([predictions.nlargest(N_STOCKS_TO_TRADE // 2),
                             predictions.nsmallest(N_STOCKS_TO_TRADE // 2)])

    todays_universe = predictions.index
    
    predictions -= 0.5 # predictions are probabilities ranging from 0 to 1
    # Setup Optimization Objective
    objective = opt.MaximizeAlpha(predictions) 

    # Setup Optimization Constraints
    constrain_gross_leverage = opt.MaxGrossLeverage(1.0)
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(-.02, .02)
    market_neutral = opt.DollarNeutral()
    # TypeError: cannot do label indexing on <class 'pandas.indexes.base.Index'> with these indexers [nan] of <type 'float'>
    sector_neutral = opt.NetPartitionExposure.with_equal_bounds(
            labels=context.risk_factors.Sector.dropna(),
            min=-0.0001,
            max=0.0001,
    )

    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
            sector_neutral,
        ],
        universe=todays_universe,
    )


def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage,
          num_positions=len(context.portfolio.positions))
 
def handle_data(context,data):
    """
    Called every minute.
    """
    pass
