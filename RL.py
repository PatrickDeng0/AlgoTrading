import util, MDP
import pandas as pd
from features import *



############################################################
# Simulate the whole process for numTrial times
# rl: the defined RL problem we have construct
# Data: daily data or else
# Mode: Only ! Sell ! and ! Buy ! available
# Learn: Whether we are training. If we are testing, set to False
def Simulate(rl, orderbooks_df, transaction_df, quantile_df, time_frame, start_time, FeatureNum, numTrial=50, learn=True):
    def buildOrder(remain, action, orderbook):
        # Find the Action, we build the order (According to the Sects[0])
        if action == 'MO':

            Order = util.SellOrder(remain, action, orderbook)
        else:
            price = orderbook.ask_prices[0] - 0.01 * int(action)
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

    orderbooks = orderbooks_df.loc[start_time : start_time + time_frame].values



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
                Sects_df = orderbooks_df.loc[start_time + time_frame-1]
                action = 'MO'
            else:

                ob_data = orderbooks[i * rl.mdp.timeGap :  (i + 1) * rl.mdp.timeGap]

                Sects = [util.OrderBook(row=ob) for ob in ob_data]
                Sects_df = orderbooks_df.loc[start_time + i * rl.mdp.timeGap: start_time + (i + 1) * rl.mdp.timeGap]

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
    # 一天太长了，考虑短一点的时间段
    timeGap = 10
    timeLevel = 5
    total_time = timeLevel * timeGap  # timeframe
    epoch=200
    train_start_time = 0
    train_end_time = total_time*epoch

    division = int((train_end_time-train_start_time)/total_time)
    print('division:',division)

    test_start_time = train_end_time
    test_end_time = test_start_time+total_time*epoch
    # orderbooks only appears here
    orderbooks = pd.read_csv('orderbook_CPE.csv')

    lag = 50
    f = all_features(orderbooks)


    '''
    quantile = get_quantile(f[['order_flow_buy', 'order_flow_sell', 'actual_spread', 'relative_spread',
       'actual_mkt_imb', 'relative_mkt_imb','liq_imb','rel_mid_trend', 'vol','eff_spread']], [0.33, 0.66])
    
    '''
    quantile = get_quantile(f[['vol']],
                            [0.33, 0.66])


    FeatureNum = len(quantile.columns)

    print('running')

    mdp = MDP.TradeMDP(totalVol=100, voLevel=5, timeLevel=timeLevel, priceFlex=4, timeGap=timeGap, FeatureLevel=3,
                       FeatureNum=FeatureNum)

    q_algo = MDP.QLearningAlgorithm(mdp, 50)

    for i in range(division):

        order_book_df = orderbooks.drop(columns=['trade_price', 'trade_size', 'trade_direction']).loc[i*total_time:(i+1)*total_time]
        transaction_df = orderbooks[['trade_price', 'trade_size', 'trade_direction']].loc[i * total_time: (i + 1) * total_time]
        Simulate(q_algo, order_book_df, transaction_df, quantile,start_time = train_start_time+i*total_time, \
             FeatureNum=FeatureNum, time_frame=total_time, numTrial=100)
        if i%10==0:
            print('This is the ',i,'th epoch')

    test_order_book_df = orderbooks.drop(columns=[ 'trade_price', 'trade_size', 'trade_direction']).loc[test_start_time:test_end_time]
    test_transaction_df = orderbooks[['trade_price', 'trade_size', 'trade_direction']].loc[test_start_time:test_end_time]


    reward = Simulate(q_algo, test_order_book_df, test_transaction_df, quantile, start_time = test_start_time \
                      ,FeatureNum=FeatureNum,time_frame=total_time*epoch, numTrial=100, learn=False)
    print(reward)

