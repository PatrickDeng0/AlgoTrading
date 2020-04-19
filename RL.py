import util

############################################################
# Simulate the whole process for numTrial times
# rl: the defined RL problem we have construct
# Data: daily data or else
# Mode: Only ! Sell ! and ! Buy ! available
# Learn: Whether we are training. If we are testing, set to False
def Simulate(rl, orderbooks, numTrial = 200, learn=True):

    def buildOrder(remain, action, orderbook):
        # Find the Action, we build the order (According to the Sects[0])
        if action == 'MO':
            Order = util.SellOrder(remain, action, orderbook)
        else:
            price = orderbook.ask_prices[0] - 0.01 * int(action)
            Order = util.SellOrder(remain, price, orderbook)
        return Order

    # Get the Reward and Newstate after taking action in Simulate with Sects
    def ActEnv(rl, iter, volume, action, orderbooks):
        # Build Order
        order = buildOrder(volume, action, orderbooks[0])

        # Simulate the Order
        for orderbook in orderbooks[1:]:
            order.SimTrade(util.OrderBook(orderbook))
        remain = volume - order.fill

        # Different Reward Calculate by Different Mode
        reward = order.turnover - start_mid * order.fill

        # Newstate Define
        if iter == rl.mdp.timeLevel:
            newState = None
        else:
            newState = rl.mdp.JudgeState([iter + 1, remain])
        return reward, newState, remain, order

    # Define the expected price
    init = util.OrderBook(orderbooks[0])
    start_mid = init.get_mid_price()

    # When we simulate a buy process for numTrial time
    for j in range(numTrial):
        remain = rl.mdp.totalVol
        state = rl.mdp.startState()
        sequences = []

        # Sects: a list contains series of orderbooks
        for i in range(rl.mdp.timeLevel + 1):
            if state is None:
                break
            if i == rl.mdp.timeLevel:
                Sects = [util.OrderBook(orderbooks[-1])]
            else:
                ob_data = orderbooks[i*rl.mdp.timeGap: (i+1)*rl.mdp.timeGap]
                Sects = [util.OrderBook(ob) for ob in ob_data]

            action = rl.getAction(state, learn)
            reward, newState, remain, pre_order = ActEnv(rl, iter=i, volume=remain, action=action,
                                                         orderbooks=Sects)
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
