
# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()   # initial position
    curr_state = start_state
    
    explored_set = []   # tracker for explored nodes
    explored_set.append(start_state)
    
    frontier = util.Stack() # new stack -> lifo
    frontier.push((start_state, [])) # first nopde intop problem set
    
    while not frontier.isEmpty() and not problem.isGoalState(curr_state):
        node, action = frontier.pop() # get deepest node
        explored_set.append(node) 
        children = problem.getSuccessors(node)  # get lsit of successors
    
        for child in children:
            coords = child[0] 
            if coords not in explored_set:
                direction = child[1] # solution
                curr_state = coords
                frontier.push((coords, action + [direction])) # add child to frontier with lsit of actions to get to it
    
    return action + [direction] # return complete list of actions to take


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
 
    explored_set = []
    explored_set.append(start_state)
    frontier = util.Queue() # new queue
    frontier.push((start_state, [])) # first nopde intop problem set
    
    while not frontier.isEmpty():
        node, action = frontier.pop() # get shallowest node
        
        if problem.isGoalState(node):   # if goal has been reached immediately, return action
            return action
        
        explored_set.append(node) # tracker for popped
        children = problem.getSuccessors(node)
        
        for child in children:
            coords = child[0]
            if not coords in explored_set:
                direction = child[1] # solution
                frontier.push((coords, action + [direction])) # add child to frontier
                explored_set.append(coords) # add child to explored set
        
    return action

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    explored_set = []
    start_state = problem.getStartState()
    frontier = util.PriorityQueue()  # priority queue holding successive states
    frontier.push((start_state, []),0)       # g(S) = 0 -> initial state
    
    while not frontier.isEmpty():
        node, action = frontier.pop() # get deepest node
        if problem.isGoalState(node):   # if deepest node is the goal
            return action
        
        if node not in explored_set:    # if node not visited yet
            explored_set.append(node) 
            children = problem.getSuccessors(node)  # get lsit of successors

            for child in children:
                coords = child[0]
                direction = child[1] # solution
                cost = problem.getCostOfActions(action+[direction])
                
                if child not in explored_set:
                    frontier.push((coords, action + [direction]), cost) # add child to frontier with lsit of actions to get to it
                else:
                    for i in range(0, len(frontier.heap)):    
                        if frontier.heap[i][0] > cost:                                     # if the node exists with a cost greater than the current node
                            frontier.heap[i] = (cost, (coords, action+[direction]))        # replace frontier node with current child  (priority, item)
                                 
                
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    explored_set = []
    start_state = problem.getStartState()
    frontier = util.PriorityQueue()  # priority queue handling successive states
    frontier.push((start_state, []), nullHeuristic(start_state, problem=problem))       # f(s) = 0 + h(s) -> initial state
    
    while not frontier.isEmpty():
        node, action = frontier.pop()
        if problem.isGoalState(node):   # if current nnode is the goal
            return action
        
        if node not in explored_set:
            explored_set.append(node)
            
            children = problem.getSuccessors(node)      # get sucessor nodes
            for child in children:                      # iterate through sucessors
                
                coords = child[0] 
                direction = child[1]    
                f_n = problem.getCostOfActions(action + [direction]) + heuristic(coords, problem)       # Total cost or f(n) = g(n) + h(n)  {cost + heuristic}
                
                if child not in explored_set:                             # if node hasn't been visited before
                    frontier.push((coords, action+[direction]), f_n)      # add child to fronter
                else:                                                                   # if node has been visited before
                    for i in range(0, len(frontier.heap)):    
                        if frontier.heap[i][0] > f_n:                                     # if the node exists with a cost greater than the current node
                            frontier.heap[i] = (f_n, (coords, action+[direction]))        # replace frontier node with current child  (priority, item)
    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
