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
def Simulate(rl, orderbooks_df, transaction_df, quantile_df, numTrial=50, learn=True):
    def buildOrder(remain, action, orderbook):
        # Find the Action, we build the order (According to the Sects[0])
        if action == 'MO':

            Order = util.SellOrder(remain, action, orderbook)
        else:
            try:
                spread = (orderbook.ask_prices[0]-orderbook.bid_prices[0])
            except:
                spread = 0
            price = orderbook.ask_prices[0] - round(spread/4 * int(action),2)
            Order = util.SellOrder(remain, price, orderbook)
        return Order

    def rank_calculation(func, orderbooks_df, quantile_df, transaction_df, need_transaction_df, *feature_name):
        # calculate the rank of the feature
        # length = len(feature_name)
        feature_rank = []

        if need_transaction_df:
            calculated_feature = func(orderbooks_df, transaction_df)
        else:
            calculated_feature = func(orderbooks_df)

        for name in feature_name:
            temp = float(calculated_feature[name].iloc[-1])
            q1, q2 = quantile_df[name].iloc[0], quantile_df[name].iloc[1]
            rank = 1 if temp <= q1 else 2 if temp <= q2 else 3
            feature_rank.append(rank)

        return feature_rank

    # Get the Reward and Newstate after taking action in Simulate with Sects
    def ActEnv(rl, iter, volume, action, orderbooks, orderbooks_df, quantile_df):
        # Build Order
        order = buildOrder(volume, action, orderbooks[0])

        # Simulate the Order
        if action != 'MO':
            for orderbook in orderbooks[1:]:
                # order.SimTrade(util.OrderBook(orderbook))

                order.SimTrade(orderbook)
        else:
            order.SimTrade(orderbooks[0])


        remain = volume - order.fill

        # Different Reward Calculate by Different Mode

        reward = order.turnover - start_mid * order.fill

        # Notice that FeatureNum !=  feature function number


        if iter != rl.mdp.timeLevel:

            feature_name = quantile_df.columns

            feature_rank = []
            '''
            feature_rank += rank_calculation(order_flow, orderbooks_df, quantile_df, transaction_df, True,
                                             *feature_name[0:6])

            feature_rank += rank_calculation(liquidity_imbalance, orderbooks_df, quantile_df, transaction_df, False,
                                             feature_name[6])

            feature_rank += rank_calculation(relative_mid_trend, orderbooks_df, quantile_df, transaction_df, False,
                                             feature_name[7])

            feature_rank += rank_calculation(volatility, orderbooks_df, quantile_df, transaction_df, False, feature_name[8])

            feature_rank += rank_calculation(effective_spread, orderbooks_df, quantile_df, transaction_df, True,
                                             feature_name[9])
            
            '''
            feature_rank += rank_calculation(volatility, orderbooks_df, quantile_df, transaction_df, False,
                                             feature_name[0])


        # Newstate Define
        if iter == rl.mdp.timeLevel:
            newState = None
        else:
            newState = rl.mdp.JudgeState([iter + 1, remain] + feature_rank)
        return reward, newState, remain, order

    orderbooks = orderbooks_df.values

    # Define the expected price
    init = util.OrderBook(row=orderbooks[0])
    start_mid = init.get_mid_price()

    # Get the list version of orderbook (instead of the dataframe version)

    # When we simulate a buy process for numTrial time
    for j in range(numTrial):
        remain = rl.mdp.totalVol
        state = rl.mdp.startState()
        sequences = []

        # Sects: a list contains series of orderbooks
        for i in range(  rl.mdp.timeLevel + 1):
            if state is None:
                break
            if i == rl.mdp.timeLevel:
                Sects = [util.OrderBook(row=orderbooks[-1])]
                Sects_df = orderbooks_df.iloc[-1]
                action = 'MO'
            else:

                ob_data = orderbooks[i * rl.mdp.timeGap :  (i + 1) * rl.mdp.timeGap]

                Sects = [util.OrderBook(row=ob) for ob in ob_data]
                Sects_df = orderbooks_df.iloc[i * rl.mdp.timeGap: (i+1) * rl.mdp.timeGap]

                action = rl.getAction(state, learn)

            reward, newState, remain, pre_order = ActEnv(rl, iter=i, volume=remain, action=action,
                                                         orderbooks=Sects, orderbooks_df=Sects_df,
                                                         quantile_df=quantile_df)

            sequences.append((state, action, reward, newState))
            state = newState

        # For update the QGrid, we update from the end to the start
        #  (for updating while using the updating results before)
        if learn:
            sequences.reverse()
            r=0
            for sequence in sequences:
                state, action, reward, newState = sequence
                rl.updateQ(state, action, reward, newState)
                r+=reward
            #print('reward:',r)

        else:
            # We are testing, so return the reward
            # Extract the reward for the sequence and sum up for the total reward in test
            return sum([sequence[2] for sequence in sequences])
        '''
        if j % 50 == 0:
            print('This is the', j, 'th iteration')
        '''
    # Explore probability upgrade if learn!
    rl.update_prob()


if __name__ == '__main__':
    # Usage: Currently, we cannot take too much features
    # If one wishes to run directly, we only apply 'volatility' and 'order_flow' in features.
    data_dir = './'
    timeGap = 10
    timeLevel = 5
    total_time = timeLevel * timeGap
    epoch=200

    # orderbooks only appears here
    orderbooks = pd.read_csv('orderbook_CPE.csv')
    lag = 50
    f = all_features(orderbooks)
    quantile = get_quantile(f[['vol']], [0.33, 0.66])
    FeatureNum = len(quantile.columns)
    order_book_df = orderbooks.drop(columns=['trade_price', 'trade_size', 'trade_direction'])
    transaction_df = orderbooks[['trade_price', 'trade_size', 'trade_direction']]

    # Split train and test epochs
    num_epochs = len(orderbooks) // total_time
    train_epochs_num = int(num_epochs * 0.9)

    # Initilize MDP
    mdp = MDP.TradeMDP(totalVol=100, voLevel=5, timeLevel=timeLevel, priceFlex=4, timeGap=timeGap, FeatureLevel=3,
                       FeatureNum=FeatureNum)
    q_algo = MDP.QLearningAlgorithm(mdp, 50)

    def spliter(df, total_time, num):
        return df.loc[num*total_time: (i+1)*total_time]

    # Training
    for i in range(train_epochs_num):
        ob_piece = spliter(order_book_df, total_time, i)
        trx_piece = spliter(transaction_df, total_time, i)
        Simulate(q_algo, ob_piece, trx_piece, quantile, numTrial=100, learn=True)

    # Test
    RL_reward = []; SL_reward = []
    for i in range(train_epochs_num, num_epochs):
        ob_piece = spliter(order_book_df, total_time, i)
        trx_piece = spliter(transaction_df, total_time, i)
        ob_all_piece = spliter(orderbooks, total_time, i)

        RL_reward.append(Simulate(q_algo, ob_piece, trx_piece, quantile, numTrial=1, learn=False))
        SL_reward.append(SL_Policy(50, ob_all_piece))

    RL_reward = np.array(RL_reward); SL_reward = np.array(SL_reward)
