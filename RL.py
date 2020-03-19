import util

############################################################
# Simulate the whole process for numTrial times
# rl: the defined RL problem we have construct
# Data: daily data or else
# Mode: Only ! Sell ! and ! Buy ! available
# Learn: Whether we are training. If we are testing, set to False
def Simulate(rl, data, numTrial = 200, learn=True):

    def buildOrder(remain, action, Sect, pre_order):
        # Find the Action, we build the order (According to the Sects[0])
        if action == 'MO':
            Order = util.SellOrder(remain, action, Sect, pre_order)
        else:
            price = Sect[10] - 0.01 * int(action)
            Order = util.SellOrder(remain, price, Sect, pre_order)
        return Order

    # Get the Reward and Newstate after taking action in Simulate with Sects
    def ActEnv(rl, iter, volume, action, Sects, pre_order=None):
        # Build Order
        order = buildOrder(volume, action, Sects[0], pre_order)

        # Simulate the Order with Sects
        order.SimTrade(Sects)
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
    start_mid = (data[0][0] + data[0][10])/2

    # When we simulate a buy process for numTrial time
    for j in range(numTrial):
        remain = rl.mdp.totalVol
        state = rl.mdp.startState()
        sequences = []
        pre_order = None
        for i in range(rl.mdp.timeLevel + 1):
            if state is None:
                break
            if i == rl.mdp.timeLevel:
                Sects = [data[-1]]
            else:
                Sects = data[i*rl.mdp.timeGap: (i+1)*rl.mdp.timeGap]

            action = rl.getAction(state, learn)
            reward, newState, remain, pre_order = ActEnv(rl, iter=i, volume=remain, action=action,
                                                         Sects=Sects, pre_order=pre_order)
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
