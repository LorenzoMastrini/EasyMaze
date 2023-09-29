class Labyrinth:
    def __init__(self, matrix=[]):
        self.matrix = matrix
        self.dimension = len(matrix) # dimension of row and columns of the matrix
        self.entrance = self.find('S')[0] # index is 0 because the entrance is only one
        self.exits = self.find('E')
        self.probabilities_list = self.probabilities()
    


    def find(self, element): # find the specified element in a matrix
        elements = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.matrix[i][j] == element:
                    elements.append((i, j))
        return elements



    def probabilities(self): #computes the distribution of some elements in the matrix (used in the heuristic funcion)
        n_goal = len(self.exits)
        occurrences = [0, 0, 0, 0, 0] # occurences of elements in a matrix: occurrences[0] counts 0s and 'X's; occurences[1] = counts the 1s; occurences[2] = counts the 2s...
        for element in range(len(occurrences)):
            for i in range(self.dimension): # iterates the lines of the matrix
                occurrences[element] += self.matrix[i].count(element)
            
        for i in range(self.dimension): #this cicle counts only the 'X'; they are considered as cost 0
            occurrences[0] += self.matrix[i].count('X')

        total = self.dimension * self.dimension
        probabilities = [0, 0, 0, 0, 0] 
        for i in range(len(occurrences)): # compute the relative frequencies of each element in the matrix
            probabilities[i] = occurrences[i] / (total - 1 - n_goal) # -1 is for the entrance 'S'

        return probabilities