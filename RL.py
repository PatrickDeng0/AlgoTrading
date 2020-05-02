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
            price = orderbook.ask_prices[0] - round(spread / 4 * int(action), 2)
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
                ob_data = orderbooks[i * rl.mdp.timeGap:  (i + 1) * rl.mdp.timeGap]
                Sects = [util.OrderBook(row=ob) for ob in ob_data]
                future_df = orderbooks_df.iloc[(i+1)*rl.mdp.timeGap + 1]

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


def main(numTrial):
    # Usage: Currently, we cannot take too much features
    # If one wishes to run directly, we only apply 'volatility' and 'order_flow' in features.
    data_dir = './'
    timeGap = 10
    timeLevel = 5

    # First timeLevel * timeGap time: time for limit order
    # Last 1 : time for market order
    total_time = timeLevel * timeGap + 1

    # orderbooks only appears here
    raw_orderbooks = pd.read_csv(data_dir + 'orderbook_new.csv')
    lag = 50
    f = all_features(raw_orderbooks, lag=lag)
    orderbooks_df = pd.concat([raw_orderbooks, f], axis=1).iloc[(lag-1):]
    orderbooks_df.index = range(len(orderbooks_df))

    quantile = get_quantile(f[['vol']], [0.33, 0.66])
    FeatureNum = len(quantile.columns)

    # Split train and test epochs
    num_epochs = len(orderbooks_df) // total_time
    train_epochs_num = int(num_epochs * 0.9)

    # Initilize MDP
    mdp = MDP.TradeMDP(totalVol=100, voLevel=5, timeLevel=timeLevel, priceFlex=4, timeGap=timeGap, FeatureLevel=3,
                       FeatureNum=FeatureNum)
    q_algo = MDP.QLearningAlgorithm(mdp, exploreStep=200, init_prob=0.8, final_prob=0.1)

    def spliter(df, total_time, num):
        return df.loc[num * total_time: (i + 1) * total_time]

    # Training
    RL_train_reward = []
    SL_train_reward = []
    for i in range(train_epochs_num):
        ob_piece = spliter(orderbooks_df, total_time, i)
        Simulate(q_algo, ob_piece, quantile, numTrial=numTrial, learn=True)
        RL_train_reward.append(Simulate(q_algo, ob_piece, quantile, numTrial=1, learn=False))
        SL_train_reward.append(SL_Policy(50, ob_piece))

    # Test
    RL_test_reward = []
    SL_test_reward = []
    for i in range(train_epochs_num, num_epochs):
        ob_piece = spliter(orderbooks_df, total_time, i)
        RL_test_reward.append(Simulate(q_algo, ob_piece, quantile, numTrial=1, learn=False))
        SL_test_reward.append(SL_Policy(50, ob_piece))

    RL_train_reward = np.array(RL_train_reward)
    RL_test_reward = np.array(RL_test_reward)
    SL_train_reward = np.array(SL_train_reward)
    SL_test_reward = np.array(SL_test_reward)
    return RL_train_reward, RL_test_reward, SL_train_reward, SL_test_reward


if __name__ == '__main__':
    RL_train_reward, RL_test_reward, SL_train_reward, SL_test_reward = main(numTrial=10)
