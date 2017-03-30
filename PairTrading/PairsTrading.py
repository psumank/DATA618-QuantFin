"""
Pairs trading algorithm.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
import math
import numpy as np
import datetime as dt
import statsmodels.tsa.stattools as ts
from scipy import stats
import statsmodels.api as sm

 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    #JBHT Vs YRCW, 
    #BAC vs C, 
    #WAL-MART Vs TARGET
    context.universe = [[sid(4108), sid(8370)],  
                      [sid(700), sid(1335)],  
                      [sid(8229), sid(21090)]]  

    context.lag = 30
    context.longPosition = False
    context.shortPosition = False
    context.selectedPair = 2
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
         
#conduct augmented dickey fuller or array x with a default
#level of 10%
def is_stationary(x, p):
    x = np.array(x)
    result = ts.adfuller(x, regression='ctt')
    #1% level
    if p == 1:
        #if DFStat <= critical value
        if result[0] >= result[4]['1%']:        #DFstat is less negative
            #is stationary
            return True
        else:
            #is nonstationary
            return False
    #5% level
    if p == 5:
        #if DFStat <= critical value
        if result[0] >= result[4]['5%']:        #DFstat is less negative
            #is stationary
            return True
        else:
            #is nonstationary
            return False
    #10% level
    if p == 10:
        #if DFStat <= critical value
        log.info('result[0]: %d'  % result[0])
        log.info('result[4]: %d'  % result[4]['10%'])
        if result[0] >= result[4]['10%']:        #DFstat is less negative
            #is stationary
            log.info("True")
            return True
        else:
            #is nonstationary
            return False
    
    
#Engle-Granger test for cointegration for array x and array y
def are_cointegrated(x, y):
    #check x is I(1) via Augmented Dickey Fuller
    x_is_I1 = not(is_stationary(x, 10))
    #check y is I(1) via Augmented Dickey Fuller
    y_is_I1 = not(is_stationary(y, 10))
    #if x and y are no stationary        
    if x_is_I1 and y_is_I1:
        b1, b0, r, p, err = stats.linregress(x,y)
        spreadHist = y - (b0 + b1 * x)
        return (ts.adfuller(spreadHist,1)[1] < 0.05)
    #if x or y are nonstationary they are not cointegrated
    else:
        return False
 
def my_rebalance(context,data):
    # assign each stock in the pair to variable:
    pair = context.universe[context.selectedPair]
    
    # get history for cointegration test:
    priceHist_X = data.history(pair[0], 'price', context.lag, '1d')
    priceHist_Y = data.history(pair[1], 'price', context.lag, '1d')
    
    #check if the pair is cointegrated.
    is_cointegrated  = are_cointegrated(priceHist_X, priceHist_Y)
    
    #Using scipy, find out the regression params.
    b1, b0, r, p, err = stats.linregress(priceHist_X,priceHist_Y)
    
    #find the spread history.
    spreadHist = priceHist_Y - ( b0 + b1 * priceHist_X )
    
    #find the spread history stand deviation & mean.
    spread_sd = spreadHist.std()
    spread_mean = spreadHist.mean()
    
    # get current price and current spread:
    price_X = data.current(pair[0],'price')
    price_Y = data.current(pair[1],'price')
    current_spread = price_Y - (b0 + b1 * price_X)
    
    #A basic strategy is:
    #If spread(t) >= Mean Spread + 2 * Standard Deviation then go Short
    #If spread(t) <= Mean Spread + 2 * Standard Deviation then go Long
    #http://gekkoquant.com/2013/01/21/statistical-arbitrage-trading-a-cointegrated-pair/
    if is_cointegrated:
        if current_spread >= spread_mean + 2 * spread_sd and not context.longPosition:
            order_target_percent(pair[0], 0.5)
            order_target_percent(pair[1], -0.5)
            context.longPosition = True
            context.shortPosition = False
        elif current_spread <= spread_mean + 2 * spread_sd and not context.shortPosition:
            order_target_percent(pair[0], -0.5)
            order_target_percent(pair[1], 0.5)
            context.longPosition = False
            context.shortPosition = True
        # else exit trade
        else:
            order_target_percent(pair[0], 0)
            order_target_percent(pair[1], 0)
            context.longPosition = False
            context.shortPosition = False
