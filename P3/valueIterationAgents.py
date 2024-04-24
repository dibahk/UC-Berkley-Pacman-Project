# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            temp_val = util.Counter()
            for state in self.mdp.getStates():
                max_val = -9999
                for action in self.mdp.getPossibleActions(state):
                    sum_val = 0.0
                    for prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        sum_val += prob[1]*(self.mdp.getReward(state, action, prob[0]) + self.discount*self.values[prob[0]])
                    max_val = max(max_val, sum_val)
                if max_val != -9999:
                    temp_val[state] = max_val
            for state in self.mdp.getStates():
                self.values[state] = temp_val[state]




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sum_val = 0.0
        for prob in self.mdp.getTransitionStatesAndProbs(state, action):
            sum_val += prob[1]*(self.mdp.getReward(state, action, prob[0]) + self.discount*self.values[prob[0]])
        return sum_val
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_val = -9999
        best_action = None
        for action in self.mdp.getPossibleActions(state):
            value = self.computeQValueFromValues(state, action)
            if value > max_val:
                best_action = action
                max_val = value
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()        
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                maximumValue = -99999
                for action in actions:
                    maximumValue = max(self.getQValue(state, action), maximumValue)
                self.values[state] = maximumValue
        states = self.mdp.getStates()


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        # in order to define the probability of each state we first define a dictionary
        predecessors = {}
        
        for state in self.mdp.getStates():
            values = []
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    # for all actions we get the probabilites and transition states
                    for prob in self.mdp.getTransitionStatesAndProbs(state,action):
                        # for new states we add them to the actionaries
                        if prob[0] in predecessors:
                            predecessors[prob[0]].add(state)
                        # for already existing states we update update them
                        else:
                            predecessors[prob[0]] = {state}
                    
                    # we add the non terminal states to the queue
                    values.append(self.computeQValueFromValues(state, action))
                # by finding the maximum q value we calculate the error and add it to the queue with negative error
                diff = abs(self.values[state] - max(values))
                pq.update(state, -diff)
        # running the algorithms for the number of iterations
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            # state with the lowest error
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                values = []
                # updating the values for each state by having the largest q 
                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(values)
            for pred in predecessors[state]:
                if not self.mdp.isTerminal(pred):
                    pred_values = []
                    for action in self.mdp.getPossibleActions(pred):
                        pred_values.append(self.computeQValueFromValues(pred, action))
                    diff = abs(self.values[pred] - max(pred_values))
                    if diff > self.theta:
                        pq.update(pred, -diff)





