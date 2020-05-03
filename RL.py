import util, MDP
import pandas as pd
from features import *
from SL import SL_Policy


############################################################
# Simulate the whole process for numTrial times
# rl: the defined RL problem we have construct
# Data: daily data or else
# Mode: Only ! Sell ! and ! Buy ! available
# Learn: Whether we are training. If we are testing, set to False
def Simulate(rl, orderbooks_df, quantile_df, numTrial=50, learn=True):
    def buildOrder(remain, action, orderbook):
        # Find the Action, we build the order (According to the Sects[0])
        if action == 'MO':
            Order = util.SellOrder(remain, action, orderbook)
        else:
            spread = orderbook.get_spread()
            price = orderbook.ask_prices[0] - round(spread / rl.mdp.priceFlex * int(action), 2)
            Order = util.SellOrder(remain, price, orderbook)
        return Order

    # Get the Reward and Newstate after taking action in Simulate with Sects
    def ActEnv(rl, iter, volume, action, orderbooks, future_df, quantile_df):
        # Build Order
        order = buildOrder(volume, action, orderbooks[0])

        # Simulate the Order
        if action != 'MO':
            for orderbook in orderbooks[1:]:
                order.SimTrade(orderbook)
        else:
            order.SimTrade(orderbooks[0])
        remain = volume - order.fill
        reward = order.turnover - start_mid * order.fill

        # Newstate Define
        if iter == rl.mdp.timeLevel:
            newState = None
        else:
            newState = rl.mdp.JudgeState([iter + 1, remain], future_df, quantile_df)
        return reward, newState, remain


    orderbooks = orderbooks_df.values
    # Define the expected price, and extract first orderbook for judge start_state
    first_orderbook_df = orderbooks_df.iloc[0]
    init = util.OrderBook(row=orderbooks[0])
    start_mid = init.get_mid_price()

    # When we simulate a sell process for numTrial time
    for j in range(numTrial):
        remain = rl.mdp.totalVol
        state = rl.mdp.startState(first_orderbook_df, quantile_df)
        sequences = []

        # Sects: a list contains series of orderbook objects
        # future_df: a pd.Series helps to judge the newState
        for i in range(rl.mdp.timeLevel + 1):
            if state is None:
                break
            elif i == rl.mdp.timeLevel:
                Sects = [util.OrderBook(row=orderbooks[-1])]
                future_df = None
            else:
                ob_data = orderbooks[i * rl.mdp.timeGap:  (i+1) * rl.mdp.timeGap]
                Sects = [util.OrderBook(row=ob) for ob in ob_data]
                future_df = orderbooks_df.iloc[(i+1) * rl.mdp.timeGap]

            action = rl.getAction(state, learn)
            reward, newState, remain = ActEnv(rl, iter=i, volume=remain, action=action, orderbooks=Sects,
                                              future_df=future_df, quantile_df=quantile_df)
            sequences.append((state, action, reward, newState))
            state = newState

        # For update the QGrid, we update from the end to the start
        #  (for updating while using the updating results before)
        if learn:
            sequences.reverse()
            for sequence in sequences:
                state, action, reward, newState = sequence
                rl.updateQ(state, action, reward, newState)

        else:
            # We are testing, so return the reward
            # Extract the reward for the sequence and sum up for the total reward in test
            return sum([sequence[2] for sequence in sequences])
    # Explore probability upgrade if learn!
    rl.update_prob()


def data_prepare(symbol, total_time):
    # orderbooks only appears here
    raw_orderbooks = pd.read_csv('./ob_{}.csv'.format(symbol))
    lag = 50
    f = all_features(raw_orderbooks, lag=lag)
    orderbooks_df = pd.concat([raw_orderbooks, f], axis=1).iloc[(lag - 1):]
    orderbooks_df.index = range(len(orderbooks_df))

    # Train test split
    num_epochs = len(orderbooks_df) // total_time
    train_num_epochs = int(num_epochs * 0.9)
    train_df = orderbooks_df.loc[:train_num_epochs*total_time]
    test_df = orderbooks_df.loc[train_num_epochs*total_time:]
    test_df.index = range(len(test_df))

    # Get quantile only from train_df
    quantile = get_quantile(train_df[['vol']], [0.33, 0.66])
    return train_df, test_df, quantile


def df_simulate(numTrial, q_algo, df, total_time, quantile, train=False):
    def spliter(df, total_time, num):
        return df.loc[num * total_time: (num + 1) * total_time]

    RL_reward = []
    SL_reward = []
    num_epochs = len(df) // total_time
    for i in range(num_epochs):
        ob_piece = spliter(df, total_time, i)
        if train:
            Simulate(q_algo, ob_piece, quantile, numTrial=numTrial, learn=True)
        RL_reward.append(Simulate(q_algo, ob_piece, quantile, numTrial=1, learn=False))
        SL_reward.append(SL_Policy(q_algo.mdp.totalVol, ob_piece))
    return np.array(RL_reward), np.array(SL_reward)


def main(symbol, numTrial):
    # Usage: Currently, we cannot take too much features
    # If one wishes to run directly, we only apply 'volatility' and 'order_flow' in features.
    totalVol = 50
    voLevel = 5
    timeGap = 10
    priceFlex = 4
    timeLevel = 5

    # First timeLevel * timeGap time: time for limit order
    # Last 1 : time for market order
    total_time = timeLevel * timeGap + 1
    train_df, test_df, quantile = data_prepare(symbol, total_time)
    FeatureNum = len(quantile.columns)

    # Initilize MDP
    mdp = MDP.TradeMDP(totalVol=totalVol, voLevel=voLevel, timeLevel=timeLevel, priceFlex=priceFlex, timeGap=timeGap,
                       FeatureLevel=3, FeatureNum=FeatureNum)
    q_algo = MDP.QLearningAlgorithm(mdp, exploreStep=2000, init_prob=0.8, final_prob=0.1)
    RL_ontrain_reward, SL_ontrain_reward = df_simulate(numTrial, q_algo, train_df, total_time, quantile, train=True)
    RL_train_reward, SL_train_reward = df_simulate(1, q_algo, train_df, total_time, quantile, train=False)
    RL_test_reward, SL_test_reward = df_simulate(1, q_algo, test_df, total_time, quantile, train=False)

    return RL_ontrain_reward, RL_train_reward, RL_test_reward, \
           SL_ontrain_reward, SL_train_reward, SL_test_reward, q_algo

if __name__ == '__main__':
    RL_ontrain_reward, RL_train_reward, RL_test_reward, \
    SL_ontrain_reward, SL_train_reward, SL_test_reward, q_algo = main(symbol='INTC', numTrial=100)
