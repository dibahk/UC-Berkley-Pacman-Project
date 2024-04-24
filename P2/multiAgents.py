# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        if action == 'Stop':
            score -= 50
        food_dist = []
        for food in newFood.asList():
            food_dist.append(manhattanDistance(newPos,food))

        ghost_dist = []
        for ghost in newGhostStates:
            ghost_dist.append(manhattanDistance(newPos,ghost.getPosition()))
        if len(food_dist) == 0:
            value = 1
        else:
            value = min(food_dist)
        return sum([score, 0.9*min(ghost_dist), -0.9*value, sum(newScaredTimes)])

    

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)
      
    def value(self,gameState,agent_i, depth):
        
        if gameState.isWin() or gameState.isLose() or depth == self.depth or gameState.getLegalActions(agent_i) == 0:
            return self.evaluationFunction(gameState)
        if agent_i == 0:
            return self.max_value(gameState,agent_i, depth)
        else:
            return self.min_value(gameState,agent_i, depth)    
        
    def max_value(self,gameState,agent_i, depth):
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(0)
        maxvalue = - math.inf 
        best_action = actions[0]
        for action in actions:
            successor= gameState.generateSuccessor(agent_i,action)
            score = self.value(successor,(agent_i+1)%num_agents, depth)           
            maxvalue = max(maxvalue,score)
            if maxvalue == score:
                best_action = action
        
        if depth == 0:
            return best_action
        return maxvalue
    
    def min_value(self,gameState, agent_i, depth):
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent_i)
        minvalue = math.inf
        next = (agent_i+1)%num_agents
        for action in actions:
            next = (agent_i+1)%num_agents
            successor= gameState.generateSuccessor(agent_i,action)
            if next == 0:
                score = self.value(successor,next,depth + 1)
            else:
                score = self.value(successor,next,depth)
                
            minvalue = min(minvalue, score)
      
        
        return minvalue
    

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action = self.value(gameState, 0, 0, - math.inf, math.inf)
        return action

    def max_value(self,gameState,agent_i, depth, alpha, beta):
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(0)
        maxvalue = - math.inf 
        best_action = actions[0]
        for action in actions:
            successor= gameState.generateSuccessor(agent_i,action)
            score, m = self.value(successor,(agent_i+1)%num_agents, depth, alpha, beta)
            maxvalue = max(maxvalue,score)
            if maxvalue == score:
                best_action = action
            alpha = max(alpha,maxvalue)
            if maxvalue > beta:
                return (maxvalue, best_action)
        return (maxvalue, best_action)
    
    def min_value(self,gameState, agent_i, depth, alpha, beta):
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent_i)
        minvalue = math.inf
        next = (agent_i+1)%num_agents
        min_action = actions[0]
        for action in actions:
            next = (agent_i+1)%num_agents
            successor= gameState.generateSuccessor(agent_i,action)
            if next == 0:
                score, m = self.value(successor,next,depth + 1, alpha, beta)
            else:
                score, m = self.value(successor,next,depth, alpha, beta)
                
            minvalue = min(minvalue, score)
            if minvalue == score:
                min_action = action
            beta = min(beta, minvalue)
            if minvalue < alpha:
                return (minvalue, min_action)        
        return (minvalue, min_action)
    
    def value(self,gameState,agent_i, depth, alpha, beta):
        
        if gameState.isWin() or gameState.isLose() or depth == self.depth or gameState.getLegalActions(agent_i) == 0:
            return self.evaluationFunction(gameState),""
        if agent_i == 0:
            return self.max_value(gameState, agent_i, depth, alpha, beta)
        else:
            return self.min_value(gameState, agent_i, depth, alpha, beta)    
        

    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)
      
    def value(self,gameState,agent_i, depth):
        
        if gameState.isWin() or gameState.isLose() or depth == self.depth or gameState.getLegalActions(agent_i) == 0:
            return self.evaluationFunction(gameState)
        if agent_i == 0:
            return self.max_value(gameState,agent_i, depth)
        else:
            return self.exp_value(gameState,agent_i, depth)    
        
    def max_value(self,gameState,agent_i, depth):
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(0)
        maxvalue = - math.inf 
        best_action = actions[0]
        for action in actions:
            successor= gameState.generateSuccessor(agent_i,action)
            score = self.value(successor,(agent_i+1)%num_agents, depth)           
            maxvalue = max(maxvalue,score)
            if maxvalue == score:
                best_action = action
        
        if depth == 0:
            return best_action
        return maxvalue
    
    def exp_value(self,gameState, agent_i, depth):
        final_score = 0.0
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent_i)
        
        len_action = float(len(actions))
        for action in actions:
            next = (agent_i + 1) % num_agents
            successor= gameState.generateSuccessor(agent_i,action)
            if next == 0:
                score = self.value(successor,next,depth + 1)
            else:
                score = self.value(successor,next,depth)

            final_score += score
        
        return final_score / len_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    food_dist = []
    for food in foods.asList():
        food_dist.append(manhattanDistance(pacmanPosition,food))

    ghost_dist = []
    for ghost in ghostPositions:
        ghost_dist.append(manhattanDistance(pacmanPosition,ghost))
    if len(food_dist) == 0:
        value = 1
    else:
        value = min(food_dist)
    remain_capsule_num = len(currentGameState.getCapsules()) + 1
    remain_food_num = currentGameState.getNumFood() + 1
    " the original "
    return sum([score, 0.9*min(ghost_dist), -0.9*value, sum(scaredTimers)])
    " the first new evaluation function "
    # return sum([10*score, 0.5*min(ghost_dist), 10.0 / value, sum(scaredTimers)])
    " the second evaluation function "
    # return sum([5*score, min(ghost_dist), 10.0 / value, 10*sum(scaredTimers), 10.0 / remain_capsule_num, 10.0 / remain_food_num])
    " the third evaluation function "
    # return sum([score, 0.8*min(ghost_dist)**0.5, -0.8*value**0.5, sum(scaredTimers)])
    
    
    

# Abbreviation
better = betterEvaluationFunction
