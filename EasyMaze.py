from Librerie.aima.search import *



class EasyMaze(Problem):
	def __init__(self, labyrinth):
		self.labyrinth = labyrinth
		super().__init__(self.labyrinth.entrance, self.labyrinth.exits)
	

	def actions(self, state):
		possible_actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

		# remove actions that can't be done based on the position of the agent in the matrix
		# with this checks we cover the cases in wich the agent is in the borders of the matrix
		if state[0] == 0:  # first row
			possible_actions.remove('UP')
			if state[1] == 0: # we are also in the first column (left-up corner)
				possible_actions.remove('LEFT')
			elif state[1] == self.labyrinth.dimension - 1: # or we are also in the last column (right-up corner)
				possible_actions.remove('RIGHT')
		elif state[0] == self.labyrinth.dimension - 1:  # last row
			possible_actions.remove('DOWN')
			if state[1] == 0: # we are also in the first column (left-down corner)
				possible_actions.remove('LEFT')
			elif state[1] == self.labyrinth.dimension - 1: # or we are also in the last column (right-down corner)
				possible_actions.remove('RIGHT')
		elif state[1] == 0: # first column
			possible_actions.remove('LEFT')
		elif state[1] == self.labyrinth.dimension - 1: #last column
			possible_actions.remove('RIGHT')
		
		possible_actions_tmp = [] # only used to delete from the original list of possible actions the actions that can't be performed without causing problems
		for action in possible_actions:
			possible_actions_tmp.append(action)
		
		# here we check all the cases where the agent is in one of the internal cells and remove actions if there is a wall
		for actions in possible_actions_tmp:
			if actions == 'UP':
				if self.labyrinth.matrix[state[0] - 1][state[1]] == 'X':
					possible_actions.remove('UP')
			if actions == 'RIGHT':
				if self.labyrinth.matrix[state[0]][state[1] + 1] == 'X':
					possible_actions.remove('RIGHT')
			if actions == 'DOWN':
				if self.labyrinth.matrix[state[0] + 1][state[1]] == 'X':
					possible_actions.remove('DOWN')
			if actions == 'LEFT':
				if self.labyrinth.matrix[state[0]][state[1] - 1] == 'X':
					possible_actions.remove('LEFT')
		
		return possible_actions


	def result(self, state, action):
		# returns the new pair of coordinates in the matrix based on the action performed
		if action == 'UP':
			new_state = (state[0] - 1, state[1]) # previous row, same column
		elif action == 'RIGHT':
			new_state = (state[0], state[1] + 1) # same row, next column
		elif action == 'DOWN':
			new_state = (state[0] + 1, state[1]) # next row, same column
		elif action == 'LEFT':
			new_state = (state[0], state[1] - 1) # same row, previous column

		return new_state
	

	def path_cost(self, c, state1, action, state2):
		cost = self.labyrinth.matrix[state2[0]][state2[1]] # check the value of the cell where the agent wants to move to
		if cost == 'E' or cost == 'S':
			return c
		elif isinstance(cost, int): # check if the value is an integer (cost of the action)
			return c + cost
		else:
			raise Exception('The cost can only be an integer value') # in case the reading of the input image introduces errors 
	

	def goal_test(self, state):
		return state in self.goal
	

	#euristica di Manhattan
	def h2(self, node):
		distances = []
		for i in range(len(self.goal)):
			distances.append(abs(node.state[0] - self.goal[i][0]) + abs(node.state[1] - self.goal[i][1]))
		return min(distances)


	def h(self, node):
		heuristics_list = []
		n_cell_list = [0, 0, 0, 0, 0] # number of cells that belong to each cost class
		for i in range(len(self.goal)): # here we iterate the calculations of the heuristic for each exit 'E' in the labyrinth
			manhattan = abs(node.state[0] - self.goal[i][0]) + abs(node.state[1] - self.goal[i][1]) - 1 # -1 because the last action doesn't have an associated cost (we are estimating the cost, not the cells)
			
			heuristic_value = 0
			for j in range(len(n_cell_list)): # j iterate the classes (cost of the cells)
				n_cell_list[j] = manhattan * self.labyrinth.probabilities_list[j] # computes how many cells has the cost j
				heuristic_value += n_cell_list[j] * j
			
			if (not isinstance(self.labyrinth.matrix[node.state[0]][node.state[1]], str)): # this adds to the estimated cost calculated at a state the cost of the state itself
				heuristic_value += self.labyrinth.matrix[node.state[0]][node.state[1]]
			heuristics_list.append(heuristic_value)
		return min(heuristics_list)



#BFS
def breadth_search_tree(problem):
  node = Node(problem.initial)

  if problem.goal_test(node.state):
    return node

  frontier = deque([node])
  explored = set()
  #print("Frontier BFS: ")
  while frontier:
    node = frontier.popleft()
    explored.add(node.state)
    for child in node.expand(problem):
      if child.state not in explored and child not in frontier: 
        if problem.goal_test(child.state):
          #print("Last Expanded: ",node)
          #print("#Explored BFS: ",len(explored))
          return child
        frontier.append(child)
        #print(child)
  return None

def breadth_search_graph(problem):
  node = Node(problem.initial)

  if problem.goal_test(node.state):
    return node

  frontier = deque([node])
  explored = set()
  #print("Frontier BFS: ")
  while frontier:
    node = frontier.popleft()
    explored.add(node.state)
    for child in node.expand(problem):
      if child.state not in explored and child not in frontier: 
        if problem.goal_test(child.state):
          #print("Last Expanded: ",node)
          #print("#Explored BFS: ",len(explored))
          return child
        frontier.append(child)
        #print(child)
  return None


#best first search (UCS)
def bestFirst_search_graph(problem, f, no_memoize=False):
	node = Node(problem.initial)

	if problem.goal_test(node.state):
		return node

	f = memoize(f, 'f')
	frontier = PriorityQueue('min', f)
	frontier.append(node)
	explored = set()

	while frontier:
		node = frontier.pop()
		if problem.goal_test(node.state):
			return node
		explored.add(node.state)
	
		for child in node.expand(problem):
			if child.state not in explored and child not in frontier: 
				frontier.append(child)
			elif child in frontier:
				inc = frontier.get_item(child)
				if f(child) < f(inc):
					del frontier[inc]
					frontier.append(child)
  	
	return None



#greedy best first (manhattan)
def greedy_best_first_search2(problem, h=None, no_memoize=False, display=False):
    h = memoize(h or problem.h2, 'h')
    return best_first_graph_search(problem, lambda n: h(n), display)


#greedy best first
def greedy_best_first_search(problem, h=None, no_memoize=False, display=False):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: h(n), display)



#A*
def astar_search(problem, h=None, no_memoize=False):
	h = memoize(h or problem.h, 'h')
	return bestFirst_search_graph(problem, lambda n: n.path_cost + h(n), no_memoize=False)

#A* (manhattan)
def astar_search2(problem, h=None, no_memoize=False):
	h = memoize(h or problem.h2, 'h')
	return bestFirst_search_graph(problem, lambda n: n.path_cost + h(n), no_memoize=False)



def show_solution(search_type, node):
	if node is None:
		print(search_type, "no solution")
	else:
		print(search_type, node.solution())
		print("Cost = ", node.path_cost)
		print("Path Length = ", node.depth)
	return