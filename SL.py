import util

############################################################
# S&L Policy:
# Submit limit price order of Expected Price at the beginning
# After certain time, deliver Market Order
def SL_Policy(totalVol, data):
    # Divide the Sects
    Sects = data[:len(data) - 1]
    init = Sects[0]
    exPrice = (init[0] + init[10])/2
    order = util.SellOrder(totalVol, round(exPrice, 2), init)
    final = data[-1]

    # Simulate the Order with Sects
    order.SimTrade(Sects)
    reward = order.turnover - exPrice * order.fill

    if not order.unfill == 0:
        MO_order = util.SellOrder(order.unfill, 'MO', final, pre_order=order)
        MO_order.SimTrade([final])
        MO_reward = MO_order.turnover - exPrice * MO_order.fill
        reward += MO_reward
    return reward
