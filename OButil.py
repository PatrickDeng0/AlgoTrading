import pandas as pd
import numpy as np
import datetime as dt
import csv
from imblearn.over_sampling import RandomOverSampler

# Global Variables
Q_TIME, Q_BID, Q_BIDSIZ, Q_ASK, Q_ASKSIZ = 0, 1, 2, 3, 4
T_TIME, T_SIZE, T_PRICE = 0, 1, 2


def t_s(time):
    t = time.split(":")
    return float(t[0]) * 3600 + float(t[1]) * 60 + float(t[2])


class OrderBook:
    def __init__(self, depth=5):
        self.depth = depth
        self.bids = {}
        self.asks = {}
        self.bid_prices = []
        self.ask_prices = []
        self.time = 0

    # Update best bid and ask price, for convenience in comparison
    # And Update the history of this order book, finally output to a csv
    def update(self):
        self.ask_prices = sorted(list(self.asks.keys()))
        self.bid_prices = sorted(list(self.bids.keys()), reverse=True)

    # Get the mid price of current order book
    # Consider if one side has 0 depth
    def get_mid_price(self):
        if len(self.ask_prices) == 0 and len(self.bid_prices) == 0:
            return 0
        elif len(self.ask_prices) == 0:
            return self.bid_prices[0]
        elif len(self.bid_prices) == 0:
            return self.ask_prices[0]
        else:
            return (self.ask_prices[0] + self.bid_prices[0]) / 2

    # Update OB due to quote
    def handle_quote(self, quote):
        self.time = quote[Q_TIME]
        # Update bids
        if quote[Q_BID] > 0:
            self.bids[quote[Q_BID]] = quote[Q_BIDSIZ] * 100
        for price in self.bid_prices:
            if price > quote[Q_BID]:
                del self.bids[price]

        # Update asks
        if quote[Q_ASK] > 0:
            self.asks[quote[Q_ASK]] = quote[Q_ASKSIZ] * 100
        for price in self.ask_prices:
            if price < quote[Q_ASK]:
                del self.asks[price]

        # Update best_price
        self.update()

    # For order book update when the trade is sell
    def sell_trade_update(self, trade_price, trade_size):
        # Sell limit order executed, now ask order book would change. Priority is descended by prices
        filled_size = 0
        for price in self.ask_prices:
            # If the price on ask order book is lower than the trade, then it must be eaten by the trade
            # So we accumulate the total numbers of orders eaten
            if price < trade_price:
                filled_size += self.asks[price]
                del self.asks[price]
            # Now if price is equal, we let the original amount of orders minus the accumulated orders
            elif price == trade_price:
                if filled_size < trade_size:
                    remain = self.asks[price] + filled_size - trade_size
                    if remain > 0:
                        self.asks[price] = remain
                    else:
                        del self.asks[price]
            else:
                break

    # For order book update when the trade is buy
    def buy_trade_update(self, trade_price, trade_size):
        # Buy limit order executed, now bid order book would change. Priority is increased by prices
        filled_size = 0
        for price in self.bid_prices:
            # If the price on ask order book is higher than the trade, then it must be eaten by the trade
            # So we accumulate the total numbers of orders eaten
            if price > trade_price:
                filled_size += self.bids[price]
                del self.bids[price]
            # Now if price is equal, we let the original amount of orders minus the accumulated orders
            elif price == trade_price:
                if filled_size < trade_size:
                    remain = self.bids[price] + filled_size - trade_size
                    if remain > 0:
                        self.bids[price] = remain
                    else:
                        del self.bids[price]
            else:
                break

    # Update OB due to trade
    def handle_trade(self, trade):
        self.time = trade[T_TIME]

        # Get the direction of this trade, and update the order book
        # direct = -1: "Sell" limit order, 1: "buy" limit order (According to Lobster)
        direct = None
        trade_price = trade[T_PRICE]
        trade_size = trade[T_SIZE]
        if len(self.ask_prices) > 0 and trade_price >= self.ask_prices[0]:
            direct = -1
            self.sell_trade_update(trade_price, trade_size)
        elif len(self.bid_prices) > 0 and trade_price <= self.bid_prices[0]:
            direct = 1
            self.buy_trade_update(trade_price, trade_size)
        else:
            pass
        # Update best_price and history
        self.update()
        return direct

    def show_order_book(self):
        def cut_depth(prices, sizes):
            pad_prices = prices.copy()
            res_sizes = [sizes[price] for price in pad_prices]
            if len(pad_prices) == 0:
                return [0 for _ in range(self.depth)], [0 for _ in range(self.depth)]
            else:
                while len(pad_prices) < self.depth:
                    pad_prices.append(pad_prices[-1])
                    res_sizes.append(0)
                return pad_prices, res_sizes

        ask_prices, ask_sizes = cut_depth(self.ask_prices, self.asks)
        bid_prices, bid_sizes = cut_depth(self.bid_prices, self.bids)
        res = []
        for i in range(self.depth):
            res.extend([ask_prices[i], ask_sizes[i],
                        bid_prices[i], bid_sizes[i]])

        # Add the mid_price of each orderbook (Using new consideration)
        res.append(self.get_mid_price())
        return np.array(res)

    def show_header(self):
        header = []
        for i in range(self.depth):
            header += ["ask_px{}".format(i + 1), "ask_sz{}".format(i + 1), "bid_px{}".format(i + 1),
                       "bid_sz{}".format(i + 1)]
        header.append('mid_price')
        return np.array(header)


def preprocess_data(quote_dir, trade_dir, out_order_book_filename, out_transaction_filename):
    print("Start pre-processing data")
    start_time = dt.datetime.now()
    df_quote = pd.read_csv(quote_dir)
    df_trade = pd.read_csv(trade_dir)

    df_quote = df_quote[['TIME_M', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ']].values
    df_trade = df_trade[['TIME_M', 'SIZE', 'PRICE']].values

    vt_s = np.vectorize(t_s)
    df_quote[:, Q_TIME] = vt_s(df_quote[:, Q_TIME])
    df_trade[:, T_TIME] = vt_s(df_trade[:, T_TIME])

    def time_selection(data):
        end_time = t_s("16:00:00")
        time_line = data[:, 0]
        return data[time_line <= end_time]

    df_quote = time_selection(df_quote)
    df_trade = time_selection(df_trade)
    n_trade = len(df_trade)
    n_quote = len(df_quote)

    order_book = OrderBook(depth=5)

    def is_quote_next(trade_idx, quote_idx):
        if df_trade[trade_idx][0] > df_quote[quote_idx][0]:
            return True
        else:
            return False

    trade_index = 0
    quote_index = 0
    transactions = []

    def add_transactions(trade_idx, direction):
        trade = df_trade[trade_idx]
        transactions.append([trade[T_PRICE], trade[T_SIZE], direction])

    def handle_trade(trade_idx, rec):
        current_trade = df_trade[trade_idx]
        _direction = order_book.handle_trade(current_trade)
        if 34200 < current_trade[0] < 57600:
            rec.writerow(order_book.show_order_book())
            add_transactions(trade_idx, _direction)

    def handle_quote(quote_idx):
        order_book.handle_quote(df_quote[quote_idx])

    with open(out_order_book_filename, 'w', newline='') as file:
        recorder = csv.writer(file, delimiter=',')
        recorder.writerow(order_book.show_header())
        while trade_index < n_trade and quote_index < n_quote:
            if is_quote_next(trade_index, quote_index):
                handle_quote(quote_index)
                quote_index += 1
            else:
                handle_trade(trade_index, recorder)
                trade_index += 1
        while trade_index < n_trade:
            handle_trade(trade_index, recorder)
            trade_index += 1
        while quote_index < n_quote:
            handle_quote(quote_index)
            quote_index += 1

    pd.DataFrame(transactions).to_csv(out_transaction_filename, header=['tx_price', 'tx_size', 'tx_direction'], index=False)

    print('Finished pre-processing data, {0:.3f} seconds'.format((dt.datetime.now() - start_time).total_seconds()))

    return


# Question 1:
# What is training dataset?
# Now we divide the dataset into separate epochs where length of epochs is the window size +1
# Window size as data input, the last line as for judge the movement of mid price
def convert_to_dataset(data_df, window_size=10, mid_price_window=5):
    data = data_df.values
    num_epochs = data.shape[0] // (window_size + mid_price_window)
    epochs_data = data[:num_epochs * (window_size + mid_price_window)]
    epochs_data = epochs_data.reshape(num_epochs, window_size + mid_price_window, epochs_data.shape[1])

    # Now epochs_data[:,:,-1] represents the mid_price of each time
    # X_all: all the X time including mid_price (under new consideration)
    # X: the output X, exclude last column from X_all
    X_all = epochs_data[:, :-mid_price_window, :]
    X = X_all[:, :, :-1]

    # Compute X moving average mid price (In the previous mid_price_window size)
    X_mid_prices = X_all[:, -mid_price_window:, -1]
    X_mid_prices = np.mean(X_mid_prices, axis=1)

    # Y: 0 for downwards, 1 for upwards
    # assuming the first and the third column are ask_price_1 and bid_price_1
    # Compute Y moving average mid price
    Y_mid_prices = epochs_data[:, -mid_price_window:, -1]
    Y_mid_prices = np.mean(Y_mid_prices, axis=1)

    Y = Y_mid_prices - X_mid_prices
    return X, Y


def over_sample(X, Y):
    # A lot of Y is 0: so actually we need 3 labels: 0 as down, 1 as remain, 2 as up
    Y_bar = Y / np.abs(Y)
    Y_bar = np.nan_to_num(Y_bar, 0).astype(int) + 1

    # Reshape X for Oversampling, and then reshape back
    X_bar = np.nan_to_num(X, 0)
    X_shape = X_bar.shape
    X_bar = X_bar.reshape((X_shape[0], -1))
    model_RandomUnderSampler = RandomOverSampler(sampling_strategy='all')
    X_bar, Y_bar = model_RandomUnderSampler.fit_sample(X_bar, Y_bar)
    X_bar = X_bar.reshape((-1, X_shape[1], X_shape[2]))

    return X_bar, Y_bar


if __name__ == '__main__':
    # testing
    data_dir = '../Data/'
    preprocess_data(data_dir + 'INTC_quote_20120621.csv', data_dir + 'INTC_trade_20120621.csv',
                    data_dir + 'orderbook.csv', data_dir + 'transaction.csv')
    # X, Y = convert_to_dataset(data_dir + 'orderbook.csv', window_size=10, mid_price_window=5)
