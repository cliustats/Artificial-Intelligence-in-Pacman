# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    newPos = successorGameState.getPacmanPosition()   # Pacman position after moving
    newFood = successorGameState.getFood()  # remaining food
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # return successorGameState.getScore()
        
    score = successorGameState.getScore()
    
    distance_To_Ghost = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])       
    if distance_To_Ghost > 0:
        ghost_distance_feature = 5.0 / distance_To_Ghost
        score -= ghost_distance_feature
        
    distance_To_Food = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
    if len(distance_To_Food) <> 0:
        food_distance_feature = 5.0 / min(distance_To_Food)
        score += food_distance_feature

    return score
    
    
    

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

  def isTerminal(self, state, currentDepth):
    return currentDepth == self.depth or state.isWin() or state.isLose()


class nMinimaxAgent(MultiAgentSearchAgent):    
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

        Directions.STOP:
          The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game
      """
      "*** YOUR CODE HERE ***"

      bestValue = float('-inf')
      bestAction = Directions.STOP             
      pacmanAction = gameState.getLegalActions(0)
            
      for action in pacmanAction:
          nextState = gameState.generateSuccessor(0, action)
          nextValue = self.Value(nextState, 0, 1)
          if nextValue > bestValue:              
              bestAction = action
              bestValue = nextValue
      return bestAction
      
   
  def Value(self, gameState, depth, agentIndex):      
      if self.isTerminal(gameState, depth):
          return self.evaluationFunction(gameState)
      elif agentIndex % gameState.getNumAgents() == 0: 
          return self.maxValue(gameState, depth, agentIndex)
      else:           
          return self.minValue(gameState, depth, agentIndex)

    
  def maxValue(self, gameState, depth, agentIndex):
      v = float('-inf')
      pacmanAction = gameState.getLegalActions(0) 
      #pacmanAction.remove('Stop')
      return max(v, (self.Value(gameState.generateSuccessor(0, action), depth, 1) for action in pacmanAction))

        
  def minValue(self, gameState, depth, agentIndex):
      v = float('inf')
      ghostAction = gameState.getLegalActions(agentIndex)
      for action in ghostAction: 
                  
          if agentIndex + 1 == gameState.getNumAgents():   # currenet agent is the last ghost, next move is pacman
              nextState = gameState.generateSuccessor(agentIndex, action)  
              v = min(v, self.Value(nextState, depth + 1, 0))   # start next depth
          else:
              nextState = gameState.generateSuccessor(agentIndex, action)
              v = min(v, self.Value(nextState, depth, agentIndex + 1)) 
          
          
      return v

    
      

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

        Directions.STOP:
          The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game
      """
      "*** YOUR CODE HERE ***"
 
      bestValue, nextAction = self.Value(gameState, 0, self.index)
      return nextAction

  def Value(self, gameState, depth, agentIndex):
      
      if self.isTerminal(gameState, depth):
          return self.evaluationFunction(gameState), None 

      if agentIndex % gameState.getNumAgents() == 0:
          return self.maxValue(gameState, depth, agentIndex % gameState.getNumAgents())
      else:      
          return self.minValue(gameState, depth, agentIndex % gameState.getNumAgents())

  def minValue(self, gameState, depth, agentIndex):   
      ghostState = gameState.getLegalActions(agentIndex)   
      nextStates = [(gameState.generateSuccessor(agentIndex, action), action) for action in ghostState]

      v = float("inf")
      bestAction = Directions.STOP

      for nextState, action in nextStates:
          nextValue, nextAction = self.Value(nextState, depth, agentIndex + 1)
          if nextValue < v:
              v = nextValue
              bestAction = action

      return v, bestAction

  def maxValue(self, gameState, depth, agentIndex):
      pacmanAction = gameState.getLegalActions(agentIndex)
      nextStates = [(gameState.generateSuccessor(agentIndex, action), action) for action in pacmanAction]

      v = float("-inf")
      bestAction = Directions.STOP


      for nextState, action in nextStates:
          nextValue, nextAction = self.Value(nextState, depth + 1, agentIndex + 1)
          if nextValue > v:
              v = nextValue
              bestAction = action

      return v, bestAction



       
    

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
    
  def getAction(self, gameState):
                
      bestValue, nextAction = self.Value(gameState, 0, self.index, float("-inf"), float("inf"))
      return nextAction

  def Value(self, gameState, depth, agentIndex, alpha, beta):
      
      if self.isTerminal(gameState, depth):
          return self.evaluationFunction(gameState), None 

      if agentIndex % gameState.getNumAgents() == 0:
          return self.maxValue(gameState, depth, agentIndex % gameState.getNumAgents(), alpha, beta)
      else:      
          return self.minValue(gameState, depth, agentIndex % gameState.getNumAgents(), alpha, beta)

  def minValue(self, gameState, depth, agentIndex, alpha, beta):   
      ghostState = gameState.getLegalActions(agentIndex)   
      nextStates = [(gameState.generateSuccessor(agentIndex, action), action) for action in ghostState]

      v = float("inf")
      bestAction = Directions.STOP

      for nextState, action in nextStates:
          nextValue, nextAction = self.Value(nextState, depth, agentIndex + 1, alpha, beta)
          if nextValue < v:
              v = nextValue
              bestAction = action
          if v < alpha:
              return v, bestAction
          beta = min(beta, v)

      return v, bestAction

  def maxValue(self, gameState, depth, agentIndex, alpha, beta):
      pacmanAction = gameState.getLegalActions(agentIndex)
      nextStates = [(gameState.generateSuccessor(agentIndex, action), action) for action in pacmanAction]

      v = float("-inf")
      bestAction = Directions.STOP

      for nextState, action in nextStates:
          nextValue, nextAction = self.Value(nextState, depth + 1, agentIndex + 1, alpha, beta)
          if nextValue > v:
              v = nextValue
              bestAction = action
          if v > beta:
              return v, bestAction
          alpha = max(alpha, v)

      return v, bestAction

   
    

    

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """      
  """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
  """
  def getAction(self, gameState):
      
      "*** YOUR CODE HERE ***"
    
      bestValue, nextAction = self.Value(gameState, 0, self.index)
      return nextAction

  def Value(self, gameState, depth, agentIndex):      
      if self.isTerminal(gameState, depth):
          return self.evaluationFunction(gameState), None 

      if agentIndex % gameState.getNumAgents() == 0:
          return self.maxValue(gameState, depth, agentIndex % gameState.getNumAgents())
      else:      
          return self.expValue(gameState, depth, agentIndex % gameState.getNumAgents())

  def expValue(self, gameState, depth, agentIndex):   
      ghostState = gameState.getLegalActions(agentIndex)   
      nextStates = [(gameState.generateSuccessor(agentIndex, action), action) for action in ghostState]

      v = 0
      bestAction = Directions.STOP
      for nextState, action in nextStates:
          nextValue, nextAction = self.Value(nextState, depth, agentIndex + 1)
          v += nextValue


      return v/len(ghostState), bestAction

  def maxValue(self, gameState, depth, agentIndex):
      pacmanAction = gameState.getLegalActions(agentIndex)
      nextStates = [(gameState.generateSuccessor(agentIndex, action), action) for action in pacmanAction]

      v = float("-inf")
      bestAction = Directions.STOP


      for nextState, action in nextStates:
          nextValue, nextAction = self.Value(nextState, depth + 1, agentIndex + 1)
          if nextValue > v:
              v = nextValue
              bestAction = action

      return v, bestAction
          
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    # NO ACTION!!!
    
    
    
    # We need to win at least
    #if currentGameState.isLose():
    #    return -1000
    #if currentGameState.isWin():
    #    return 1000
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    score = currentGameState.getScore()
   
    # distance to ghosts
    for ghost in newGhostStates:
        distance_To_Ghost = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if distance_To_Ghost > 0:
            if ghost.scaredTimer > 0:   
                scared_ghost_feature = 100.0 / distance_To_Ghost   # heavy weight for eating a scared ghost!
                score += scared_ghost_feature 
            else: 
                non_scared_ghost_feature = 10.0 / distance_To_Ghost   # if ghosts are not scared, we are scared!
                score -= non_scared_ghost_feature  

    # distance to closest food
    distance_To_Food = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
    if len(distance_To_Food) <> 0:
        food_distance_feature = 10.0 / min(distance_To_Food)
        score += food_distance_feature

    return score


    """
    score = successorGameState.getScore()
    
    distance_To_Ghost = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])       
    if distance_To_Ghost > 0:
        ghost_distance_feature = 5.0 / distance_To_Ghost
        score -= ghost_distance_feature
        
    distance_To_Food = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
    if len(distance_To_Food) <> 0:
        food_distance_feature = 5.0 / min(distance_To_Food)
        score += food_distance_feature

    return score
    """


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

