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


import math
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        foodDistances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min(foodDistances) if foodDistances else 0

        score += 99 if successorGameState.hasFood(newPos[0], newPos[1]) else score

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            scaredTime = ghostState.scaredTimer
            ghostDistance = util.manhattanDistance(newPos, ghostPos)
            if scaredTime == 0 and ghostDistance <= 1:
                score -= 1000  

        score -= minFoodDistance**1e-2

        return score, successorGameState.getScore()
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(gameState, depth, index):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            legalActions = gameState.getLegalActions(index)

            if index == 0:
                bestValue = float("-inf")
                bestAction = None
                for action in legalActions:
                    successor = gameState.generateSuccessor(index, action)
                    value, _ = minimax(successor, depth, index + 1)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:
                bestValue = float("inf")
                bestAction = None
                idx = index + 1
                if idx >= gameState.getNumAgents():
                    idx = 0
                    depth -= 1
                for action in legalActions:
                    successor = gameState.generateSuccessor(index, action)
                    value, _ = minimax(successor, depth, idx)
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction

        value, action = minimax(gameState, self.depth, index=0)
        return action
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(gameState, depth, index, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            legalActions = gameState.getLegalActions(index)

            if index == 0:
                bestValue = float("-inf")
                bestAction = None
                for action in legalActions:
                    successor = gameState.generateSuccessor(index, action)
                    value, _ = alphabeta(successor, depth, index + 1, alpha, beta)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                    if beta < value:
                        # break
                        return value, action
                    alpha = max(alpha, bestValue)
                return bestValue, bestAction
            else:
                bestValue = float("inf")
                bestAction = None
                idx = index + 1
                if idx >= gameState.getNumAgents():
                    idx = 0
                    depth -= 1
                for action in legalActions:
                    successor = gameState.generateSuccessor(index, action)
                    value, _ = alphabeta(successor, depth, idx, alpha, beta)
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                    if value < alpha:
                        # break
                        return value, action
                    beta = min(beta, bestValue)
                return bestValue, bestAction

        value, action = alphabeta(gameState, self.depth, 0, float("-inf"), float("inf"))
        return action
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, depth, index):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            legalActions = gameState.getLegalActions(index)

            if index == 0:
                bestValue = float("-inf")
                bestAction = None
                for action in legalActions:
                    successor = gameState.generateSuccessor(index, action)
                    value, _ = expectimax(successor, depth, index + 1)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:
                totalValue = 0
                idx = index + 1
                if idx >= gameState.getNumAgents():
                    idx = 0
                    depth -= 1
                for action in legalActions:
                    successor = gameState.generateSuccessor(index, action)
                    value, _ = expectimax(successor, depth, idx)
                    totalValue += value
                expValue = totalValue / len(legalActions)
                return expValue, None

        value, action = expectimax(gameState, self.depth, index=0)
        return action
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    - penalty by Remaining food count
    - ecourage by Distance to the nearest food pellet
    - Distance to the nearest ghost (scared or not)
    - ecourage Pacman to be close to the power pellet
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    # print("start")

    remainingFoodCount = len(foodList)
    # print(remainingFoodCount)
    score -= remainingFoodCount * 5
    # print("remainingFoodCount", score)

    if foodList:
        minFoodDistance = min([util.manhattanDistance(pacmanPosition, food) for food in foodList])
        score -= minFoodDistance ** 1e-3
        # print("nearestFoodDistance", score)
    else:
        minFoodDistance = 0

    minGhostDistance = min([util.manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates])
    score +=  minGhostDistance if  minGhostDistance <= 1 else score
    # print("nearestGhostDistance", score)

    score += 200  if any(scaredTimes) else score
    # print("scaredTimes", score)

    for capsule in capsules:
        capsuleDistance = util.manhattanDistance(pacmanPosition, capsule)
        score += 100 if capsuleDistance <= 1 else score
    # print("capsules", score)

    return score
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
