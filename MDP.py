import random


######################################
# Construct MDP here!
class TradeMDP:
    # totalVol: total volume of the stocks we want to trade (Changeable)
    # voLevel: the level of remaining inventory (Coefficient)
    # timeLevel: total operation times (Coefficient)
    # priceFlex: the range of price we could bid or ask (Coefficient)
    # timeGap: the number
    #  of ticks between two operations (Const)
    # states: the whole set of states the MDP could have
    # T_states: the states divided by the time level
    # VolState: 0:None; 1:1-eachVol; 2:eachVol+1 - 2eachVol ...
    def __init__(self, totalVol, voLevel, timeLevel, priceFlex, timeGap):
        self.totalVol = totalVol
        self.voLevel = voLevel
        self.timeLevel = timeLevel
        self.priceFlex = priceFlex
        self.timeGap = timeGap
        self.eachVol = totalVol // voLevel
        self.states = []
        self.T_states = []
        for i in range(self.timeLevel + 1):
            T_state = []
            for j in range(1, self.voLevel + 1):
                self.states.append((i, j))
                T_state.append((i, j))
            self.T_states.append(T_state)

        # For terminal state, when the remain is 0, the state is None
        self.states.append(None)

    def startState(self):
        digit = (0, self.totalVol)
        return self.JudgeState(digit)

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def Actions(self, state):
        # action of terminal state is None
        if state is None:
            return [None]
        # Market Order when we have no time
        elif state[0] == self.timeLevel:
            return ['MO']
        else:
            return [str(i) for i in range(-self.priceFlex, self.priceFlex+1)]

    # Given the accurate digit, we decide the state it is in
    def JudgeState(self, digit):
        if int(digit[1]) == 0:
            return
        volState = ((digit[1]-1) // self.eachVol) + 1
        return digit[0], int(volState)


############################################################
# Q-learning Algo of a certain MDP problem
# explorationProb: the epsilon value indicating how frequently the policy returns a random action
class QLearningAlgorithm:
    def __init__(self, mdp, exploreStep, explorationProb=0.2, ):
        self.mdp = mdp
        self.init_prob = explorationProb
        self.explorationProb = explorationProb
        self.numIters = {}
        self.QGrid = {}
        for state in mdp.states:
            for action in mdp.Actions(state):
                self.numIters[(state, action)] = 0
                self.QGrid[(state, action)] = 0
        # Increase only in simulation, and exploration prob changes
        self.totalIter = 0
        self.explorationStep = exploreStep
        self.final_prob = 0.01

    # Return the Q function
    def getQ(self, state, action):
        return self.QGrid[(state, action)]

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state, learn):
        if state is None:
            return
        elif state[0] == self.mdp.timeLevel:
            return 'MO'
        elif learn and random.random() < self.explorationProb:
            return random.choice(self.mdp.Actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.mdp.Actions(state))[1]

    # Call this function to get the step size to update the Q value
    def getStepSize(self, state, action):
        return 1.0 / (self.numIters[(state, action)] + 1)

    # We will call this function with (s, a, r, s'), update the Q-Grid and number of each state being visited
    # Note that if s is a terminal state, then s' will be None.
    def updateQ(self, state, action, reward, newState):
        sample = max([self.getQ(newState, next_act) for next_act in self.mdp.Actions(newState)])\
                 + reward - self.getQ(state, action)
        self.numIters[(state, action)] += 1
        self.QGrid[(state, action)] += self.getStepSize(state, action) * sample

    def update_prob(self):
        # Exploration prob decay as number of episodes increases, at 1/rl.explorationStep per step
        # (bigger than rl.final_prob)
        self.totalIter += 1
        self.explorationProb = max(self.final_prob, self.init_prob * (1 - self.totalIter / self.explorationStep))
