import util
import pandas as pd
import numpy as np

############################################################
# S&L Policy:
# Submit limit price order of Expected Price at the beginning
# After certain time, deliver Market Order
# Orderbooks: A list of orderbook rows
def SL_Policy(totalVol, orderbook_df):
    orderbooks = orderbook_df.values
    init = util.OrderBook(row=orderbooks[0])
    exPrice = round(init.get_mid_price(), 2)
    order = util.SellOrder(totalVol, exPrice, init)

    # Simulate the Order
    for orderbook in orderbooks[1:-1]:
        order.SimTrade(util.OrderBook(row=orderbook))
        if order.termin:
            break
    reward = order.turnover - exPrice * order.fill

    final = util.OrderBook(row=orderbooks[-1])
    if not order.termin:
        MO_order = util.SellOrder(order.unfill, 'MO', final)
        MO_order.SimTrade(final)
        MO_reward = MO_order.turnover - exPrice * MO_order.fill
        reward += MO_reward
    return reward


if __name__ == '__main__':

    # Usage: read orderbook data (depth of 5 or full depth, both okay)
    data_dir = '../Data/'
    orderbooks = pd.read_csv(data_dir + 'orderbook.csv')
    demo = orderbooks.loc[:5].values
    reward = SL_Policy(100, demo)
    print(reward)
