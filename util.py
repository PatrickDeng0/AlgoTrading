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
    def __init__(self, depth=5, row=None):
        self.time = 0
        self.depth = depth
        self.bids = {}
        self.asks = {}
        self.bid_prices = []
        self.ask_prices = []
        self.trade_price = np.nan
        self.trade_size = 0

        if row is not None:
            self.load_orderbook(row)

    # Reconstruct the object of OrderBook from a np.array that contain a single row of full orderbook
    def load_orderbook(self, row):
        self.time = row[0]
        self.depth = (len(row) - 4) // 4

        # Get the locations of asks and bids, then reconstruct the orderbook (Only if size > 0)
        ask_prices = np.arange(self.depth) * 4 + 1
        bid_prices = np.arange(self.depth) * 4 + 3
        for ask_price in ask_prices:
            if row[ask_price + 1] > 0:
                self.asks[row[ask_price]] = row[ask_price + 1]
        for bid_price in bid_prices:
            if row[bid_price + 1] > 0:
                self.bids[row[bid_price]] = row[bid_price + 1]
        self.update()

        self.trade_price = row[-2]
        self.trade_size = row[-1]

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

        # Update best_price and trade information
        self.update()
        self.trade_price = np.nan
        self.trade_size = 0

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
        # Update trade information
        direct = None
        self.trade_price = trade_price = trade[T_PRICE]
        self.trade_size = trade_size = trade[T_SIZE]
        if len(self.ask_prices) > 0 and trade_price >= self.ask_prices[0]:
            direct = -1
            self.sell_trade_update(trade_price, trade_size)
        elif len(self.bid_prices) > 0 and trade_price <= self.bid_prices[0]:
            direct = 1
            self.buy_trade_update(trade_price, trade_size)
        else:
            pass

        # Update best_price
        self.update()
        return direct

    # Full: show the full depth of orderbook, or depth = self.depth
    def show_order_book(self, trade_direction, full=False):
        def cut_depth(prices, sizes, depth):
            pad_prices = prices.copy()
            res_sizes = [sizes[price] for price in pad_prices]
            if len(pad_prices) == 0:
                return [0 for _ in range(depth)], [0 for _ in range(depth)]
            else:
                while len(pad_prices) < depth:
                    pad_prices.append(pad_prices[-1])
                    res_sizes.append(0)
                return pad_prices, res_sizes

        if full:
            show_depth = max(len(self.ask_prices), len(self.bid_prices))
        else:
            show_depth = self.depth

        ask_prices, ask_sizes = cut_depth(self.ask_prices, self.asks, show_depth)
        bid_prices, bid_sizes = cut_depth(self.bid_prices, self.bids, show_depth)
        res = [self.time]
        for i in range(show_depth):
            res.extend([ask_prices[i], ask_sizes[i],
                        bid_prices[i], bid_sizes[i]])

        # Add the mid_price of each orderbook (Using new consideration)
        res.extend([self.get_mid_price(), self.trade_price, self.trade_size, trade_direction])
        return np.array(res)

    def show_header(self):
        header = ['time']
        for i in range(self.depth):
            header += ["ask_px{}".format(i + 1), "ask_sz{}".format(i + 1), "bid_px{}".format(i + 1),
                       "bid_sz{}".format(i + 1)]
        header.extend(['mid_price', 'trade_price', 'trade_size', 'trade_direction'])
        return np.array(header)


class SellOrder:
    def __init__(self, volume, price, orderbook):
        self.totalVol = volume
        self.price = price
        self.unfill = volume
        self.fill = 0
        self.turnover = 0
        
        # Compute the rank in the queue, initialized as None
        self.rank = None
        self.termin = False
        self.queue_update(orderbook)

    def queue_update(self, orderbook):
        # Market Order have 0 rank; or if we have been in the front of the queue, we keep rank 0
        if self.termin:
            return
        elif isinstance(self.price, str) or self.rank == 0:
            self.rank = 0
        else:
            # Not market order, then we compute the rank
            newrank = orderbook.asks.get(self.price, 0)
            if self.rank is None:
                self.rank = newrank
            else:
                self.rank = min(newrank, self.rank)

    # Given the trade price and size for our order, update the status
    def updateStatus(self, price, size):
        self.turnover += price * size
        self.fill += size
        self.unfill = self.totalVol - self.fill
        self.termin = (self.unfill == 0)
        
    def SimTrade(self, orderbook):
        # Condition 1: Judge whether the order is filled or not.
        if self.termin:
            return

        # If we are not Market Order,
        if not isinstance(self.price, str):
            if not np.isnan(orderbook.trade_price):
                # There is trade happening, and our price is better than the trade price, then we could trade
                # The trade price for us will be the same as trade price
                # But the amount is up to 10% of the total trade size for less market impact in our model
                size = 0
                if self.price < orderbook.trade_price:
                    self.rank = 0
                    size = min(orderbook.trade_size * 0.1, self.unfill)

                # If our price is the same as trade price, then we have to consider our rank in the queue
                elif self.price == orderbook.trade_price:
                    # Firstly, our rank must be decrease. If we are at the front, then we could trade
                    remain = self.rank - orderbook.trade_size
                    self.rank = max(0, remain)
                    if remain < 0:
                        size = min(-remain * 0.1, self.unfill)
                self.updateStatus(orderbook.trade_price, size)

            # After matching trade, or there is no trade, then we could only update rank
            self.queue_update(orderbook)

        # For market order, simply cross through the spread
        else:
            for bid_price in orderbook.bid_prices:
                size = min(orderbook.bids[bid_price], self.unfill)
                self.updateStatus(bid_price, size)
                if self.termin:
                    return
            # Considering market order failure:
            # If there is not enough bid orders to eat, then we have to force all the trade at ask1 price
            self.updateStatus(orderbook.ask_prices[0], self.unfill)


def preprocess_data(quote_dir, trade_dir, out_order_book_filename):
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

    def handle_trade(trade_idx, rec):
        current_trade = df_trade[trade_idx]
        _direction = order_book.handle_trade(current_trade)
        if 34200 < current_trade[0] < 57600:

            # Notice that if use full depth, there might be errors in the columns names
            # Should take care!
            rec.writerow(order_book.show_order_book(_direction, full=False))

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

    print('Finished pre-processing data, {0:.3f} seconds'.format((dt.datetime.now() - start_time).total_seconds()))

    return


if __name__ == '__main__':
    # testing
    data_dir = '../Data/'
    preprocess_data(data_dir + 'INTC_quote_20120621.csv', data_dir + 'INTC_trade_20120621.csv',
                    data_dir + 'orderbook.csv')
