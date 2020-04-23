import numpy as np
import pandas as pd
import time

"""
All feature functions assume the same number of rows in order book and trades
i.e. order book only updates upon new trades
Order book columns: ask_px1, ask_sz1, bid_px1, bid_sz1, ask_px2, ask_sz2, bid_px2, bid_sz2..., mid_price
Transactions columns: trade_price, trade_size, trade_direction (-1 for sell, 1 for buy, na for mid price transactions)
"""


def div(nu, de, alpha=1):
    p = pd.Series(np.zeros_like(de))
    p[:] = alpha
    return nu / pd.concat([de, p], axis=1).max(axis=1)


def mid(order_book_df):
    return order_book_df['mid_price']


def order_flow(order_book_df, transaction_df, lag=50):
    """
    order flow = the ratio of the volume of market buy(sell) orders arriving in the prior n observations
                 to the resting volume of ask(bid) limit orders at the top of book
    This feature is constructed according to the paper.
    Intuition: an increase in this ratio will more likely deplete the best ask level and the mid-price will up-tick,
               and vice-versa for a down-tick.
    actual spread = ask_price_1 - bid_price_1
    relative spread =  (actual spread / mid price) * 10000
    actual market imbalance = the volume of market buy orders arriving in the prior n observations
                            - the volume of market sell orders arriving in the prior n observations
    This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
    relative market imbalance = actual market imbalance / actual spread
    This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
    Intuition: a small actual spread combined with a strongly positive actual market imbalance
               would indicate buying pressure.
    """
    flow = pd.concat([order_book_df[['ask_sz1', 'bid_sz1']], transaction_df[['trade_size', 'trade_direction']]], axis=1)
    flow['buy_vol'] = 0
    flow['sell_vol'] = 0
    flow.loc[flow['trade_direction'] == 1, 'buy_vol'] = flow['trade_size']
    flow.loc[flow['trade_direction'] == -1, 'sell_vol'] = flow['trade_size']
    flow['order_flow_buy'] = div(flow['buy_vol'].rolling(lag).sum(), flow['ask_sz1'])
    flow['order_flow_sell'] = div(flow['sell_vol'].rolling(lag).sum(), flow['bid_sz1'])

    mid_price = mid(order_book_df)
    flow['actual_spread'] = order_book_df['ask_px1'] - order_book_df['bid_px1']
    flow['relative_spread'] = (flow['actual_spread'] / mid_price) * 1000

    flow['actual_mkt_imb'] = \
        flow['buy_vol'].rolling(lag, min_periods=1).sum() - flow['sell_vol'].rolling(lag, min_periods=1).sum()
    flow['relative_mkt_imb'] = div(flow['actual_spread'], flow['actual_mkt_imb'])

    return flow.drop(columns=['ask_sz1', 'bid_sz1', 'trade_size', 'trade_direction', 'buy_vol', 'sell_vol'])


def liquidity_imbalance(order_book_df):
    """
    liquidity imbalance at level i = ask_vol_i / (ask_vol_i + bid_vol_i)
    This feature is constructed according to the ppt.
    """
    liq_imb = {}
    for i in range(1, int(1 + order_book_df.shape[1] / 4)):
        a, b = order_book_df['ask_sz{}'.format(i)], order_book_df['bid_sz{}'.format(i)]
        liq_imb['liq_imb_{}'.format(i)] = a / (a + b)

    return pd.DataFrame(liq_imb)


def relative_mid_trend(order_book_df):
    """
    First, construct a variation on mid-price where the average of the bid and ask prices is weighted
    according to their inverse volume. Then, divide this variation by common mid price.
    This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P10
    Intuition: a larger relative_mid_price_trend would more likely lead to a up-tick.
    """

    nom = div(order_book_df['ask_px1'], order_book_df['ask_sz1'])
    nom += div(order_book_df['bid_px1'], order_book_df['bid_sz1'])
    den = div(1, order_book_df['ask_sz1']) + div(1, order_book_df['bid_sz1'])
    mid_price_inv_vol_weighted = div(nom, den)
    mid_price = mid(order_book_df)

    return pd.DataFrame({'rel_mid_trend': mid_price_inv_vol_weighted / mid_price})


def volatility(order_book_df, lag=50):
    """
    The volatility is the standard deviation of the last n mid prices returns then divided by 100
    This feature is derived from paper: Angelo Ranaldo..._Order aggressiveness in limit order book markets...P4
    """
    mid_price = mid(order_book_df)
    mid_price_return = mid_price.shift(-1) - mid_price
    volatility_look_ahead = (mid_price_return.rolling(lag, min_periods=1).std()) / 100
    return pd.DataFrame({'vol': volatility_look_ahead.shift(1)})


def aggressiveness(order_book_df, transaction_df, lag=50):
    """
    bid(ask) limit order aggressiveness = the ratio of bid(ask) limit orders submitted at no lower(higher) than
                                                       the best bid(ask) prices in the prior n observations
                                                    to total bid(ask) limit orders submitted in prior 50 observations
    This feature is derived from book: Irene Aldridge_High-frequency trading...(2013) P186
    Intuition: The higher the ratio, the more aggressive is the trader in his bid(ask) to capture the best
               available price and the more likely the trader is to believe that the price is about to
               move away from the mid price.
    """
    df = pd.concat([order_book_df[['ask_px1', 'bid_px1']], transaction_df], axis=1)

    is_aggr_sell = (df['trade_direction'] == -1) & (df['trade_price'] <= df['ask_px1'].shift(1))
    df['aggr_sell_size'] = 0
    df.loc[is_aggr_sell, 'aggr_sell_size'] = df['trade_size']
    df['sell_tx_size'] = 0
    df.loc[df['trade_direction'] == -1, 'sell_tx_size'] = df['trade_size']
    aggr_sell_ratios = div(
        df['aggr_sell_size'].rolling(lag, min_periods=1).sum(), df['sell_tx_size'].rolling(lag, min_periods=1).sum())

    is_aggr_buy = (df['trade_direction'] == 1) & (df['trade_price'] >= df['bid_px1'].shift(1))
    df['aggr_buy_size'] = 0
    df.loc[is_aggr_buy, 'aggr_buy_size'] = df['trade_size']
    df['buy_tx_size'] = 0
    df.loc[df['trade_direction'] == -1, 'buy_tx_size'] = df['trade_size']
    aggr_buy_ratios = div(
        df['aggr_buy_size'].rolling(lag, min_periods=1).sum(), df['buy_tx_size'].rolling(lag, min_periods=1).sum())

    return pd.DataFrame({'aggr_b_ratios': aggr_buy_ratios, 'aggr_sell_ratios': aggr_sell_ratios})


def effective_spread(order_book_df, transaction_df):
    """
    The effective spread is computed as difference between the latest trade price and mid price
                                        divided by mid price, then times 1000.
    This feature is derived from book: Irene Aldridge_High-frequency trading...(2013) P191
    Intuition: The effective spread measures how far, in percentage terms, the latest realized price
               fell away from the simple mid price.
    """
    mid_price = mid(order_book_df)
    return pd.DataFrame({'eff_spread': (transaction_df['trade_price'] / mid_price - 1) * 1000})


def illiquidity(order_book_df, lag=50):
    """
    The illiquidity is computed as the ratio of absolute stock return to its dollar volume.
    This feature is derived from Amihud (2002)
    """
    mid_price = mid(order_book_df)
    mid_price_ret = np.log(mid_price) - np.log(mid_price.shift(1))
    ret_over_volume = abs(mid_price_ret) / (order_book_df['ask_sz1'] + order_book_df['bid_sz1'])
    return pd.DataFrame({'illiquidity': ret_over_volume.rolling(lag, min_periods=1).sum()})


def relative_vol(order_book_df, lag=50):
    """
    Relative volume is computed as the ratio of current volume to the historical average volume
    """
    rel_vol = {}
    for i in range(1, int(1 + order_book_df.shape[1] / 4)):
        rel_ask_sz = \
            order_book_df['ask_sz{}'.format(i)] / order_book_df['ask_sz{}'.format(i)].rolling(lag, min_periods=1).mean()
        rel_vol['rel_ask_sz{}'.format(i)] = rel_ask_sz
        rel_bid_sz = \
            order_book_df['bid_sz{}'.format(i)] / order_book_df['bid_sz{}'.format(i)].rolling(lag, min_periods=1).mean()
        rel_vol['rel_bid_sz{}'.format(i)] = rel_bid_sz

    return pd.DataFrame(rel_vol)


def volume_depth(order_book_df):
    """
    Volume depth is computed as the ratio of best volume to the sum of all depth volume
    """
    n = len(order_book_df)
    total_ask, total_bid = np.zeros(n), np.zeros(n)
    vol_depth = {}
    for i in range(1, int(1 + order_book_df.shape[1] / 4)):
        total_ask += order_book_df['ask_sz{}'.format(i)]
        total_bid += order_book_df['bid_sz{}'.format(i)]
    vol_depth['ask_vol_depth'] = order_book_df['ask_sz1'] / total_ask
    vol_depth['bid_vol_depth'] = order_book_df['bid_sz1'] / total_bid
    return pd.DataFrame(vol_depth)


def volume_rank(order_book_df, lag=50):
    """
    volume rank is computed as the rank of current volume with respect to the previous n days volume
    """

    def roll_rank(x):
        return (x.argsort().argsort()[-1] + 1.0) / len(x)

    vol_rank = {}
    for i in range(1, int(1 + order_book_df.shape[1] / 4)):
        rank_ask_vol = order_book_df['ask_sz{}'.format(i)].rolling(lag, min_periods=1).apply(roll_rank, raw=True)
        rank_bid_vol = order_book_df['bid_sz{}'.format(i)].rolling(lag, min_periods=1).apply(roll_rank, raw=True)

        rank_ask_vol = rank_ask_vol.fillna(method='ffill', axis=0)
        rank_bid_vol = rank_bid_vol.fillna(method='ffill', axis=0)
        rank_ask_vol = np.clip(rank_ask_vol, 0, 1)
        rank_bid_vol = np.clip(rank_bid_vol, 0, 1)

        vol_rank['rank_bid_sz{}'.format(i)] = rank_bid_vol
        vol_rank['rank_ask_sz{}'.format(i)] = rank_ask_vol

    return pd.DataFrame(vol_rank)


def ask_bid_correlation(order_book_df, lag=50):
    """
    ask bid volume correlation is computed as 50 days time series correlation between ask and bid volume for each level
    """
    ask_bid_corr = {}
    for i in range(1, int(1 + order_book_df.shape[1] / 4)):
        corr_sz = order_book_df['ask_sz{}'.format(i)].rolling(lag, min_periods=1) \
            .corr(order_book_df['bid_sz{}'.format(i)]).fillna(method='ffill', axis=0)
        corr_sz = np.clip(corr_sz, -1, 1)
        ask_bid_corr['corr_sz{}'.format(i)] = corr_sz

    return pd.DataFrame(ask_bid_corr)


def technical_indicators(mid_price):
    tech = {}
    n = len(mid_price)

    tech['mid_ma7'] = mid_price.rolling(7, min_periods=1).mean()
    tech['mid_ma21'] = mid_price.rolling(21, min_periods=1).mean()

    dma = mid_price.rolling(10, min_periods=1).mean() - mid_price.rolling(50, min_periods=1).mean()
    ama = dma.rolling(10, min_periods=1).mean()
    tech['dma'] = dma
    tech['ama'] = ama

    k = 2 / (20 + 1)
    tech['ema'] = mid_price * k + mid_price.shift(1) * (1 - k)

    length = 12
    psy_value = np.full(n, 50.0)  # the PSY values for first {length} days are set to be 50
    for i in range(length, n):
        psy_value[i] = sum(np.array(mid_price[i - length + 1:i + 1]) > np.array(mid_price[i - length:i])) / length * 100
    tech['psy'] = psy_value

    roc = (mid_price - mid_price.shift(length)) / mid_price.shift(length) * 100
    tech['roc'] = roc.fillna(0)

    length = 14
    diff = (mid_price - mid_price.shift(1)).fillna(0)
    tech['rsi'] = diff.rolling(length, min_periods=1) \
        .apply(lambda x: sum(x[x > 0]) / sum(abs(x)) * 100, raw=True) \
        .shift(1) \
        .fillna(50)
    tech['cmo'] = diff.rolling(length, min_periods=1) \
        .apply(lambda x: (sum(x[x > 0]) / sum(abs(x)) - 1) * 100, raw=True) \
        .shift(1) \
        .fillna(50)

    length = 6
    tech['bias'] = \
        ((mid_price - mid_price.rolling(length, min_periods=1).mean()) /
         mid_price.rolling(length, min_periods=1).mean() * 100).fillna(0)

    length = 20
    width = 2
    mid_line = mid_price.rolling(length, min_periods=1).mean()
    tech['upper_line'] = mid_line + width * mid_price.rolling(length, min_periods=1).std().shift(1)
    tech['lower_line'] = mid_line - width * mid_price.rolling(length, min_periods=1).std().shift(1)

    tech['fft'] = np.abs(np.fft.fft(mid_price))

    return pd.DataFrame(tech)


def all_features(ob, lag=50):
    start_t = time.time()
    print("Start creating features from order books and transactions data")
    order_book_df = ob.drop(columns=['time', 'trade_price', 'trade_size', 'trade_direction'])
    transaction_df = ob[['trade_price', 'trade_size', 'trade_direction']]
    features = [
        order_flow(order_book_df, transaction_df, lag),
        liquidity_imbalance(order_book_df),
        relative_mid_trend(order_book_df),
        volatility(order_book_df, lag),
        aggressiveness(order_book_df, transaction_df, lag),
        effective_spread(order_book_df, transaction_df),
        illiquidity(order_book_df, lag),
        relative_vol(order_book_df),
        volume_depth(order_book_df),
        volume_rank(order_book_df),
        ask_bid_correlation(order_book_df, lag),
        technical_indicators(mid(order_book_df))
    ]
    print("Finished creating features, time lapse: {0:.3f} seconds".format(time.time() - start_t))
    return pd.concat(features, axis=1)


if __name__ == "__main__":
    order_book_filename = './data/order_book.csv'
    o = pd.read_csv(order_book_filename)
    lag = 50
    f = all_features(o)
    f.to_csv("./data/raw_features.csv")
