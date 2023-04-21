import time
from EasyMaze import *
from Labyrinth import Labyrinth
from Librerie.aima.search import *
import cv2



img = cv2.imread('prova.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('Image', thresh1)
cv2.waitKey(0)



l = Labyrinth(( ( 0 , 0 ,'S', 1 , 1 , 1 , 1 ),
	            ( 1 ,'X', 4 ,'X','X', 0 ,'X'),
                ( 1 , 0 , 1 ,'X', 0 ,'X','X'),
	            ('X','X', 1 , 1 , 4 ,'X','X'),
			    ( 1 , 1 , 1 ,'X', 4 ,'X', 1 ),
			    ('X','X', 0 , 0 , 0 ,'X', 1 ),
			    ( 0 , 0 ,'E','X', 1 , 1 ,'E')))
p1 = EasyMaze(l)
#print("Azioni possibili: " +  str(p1.actions((2, 2))))
#print("Result: " +  str(p1.result(state=(2, 2), action='RIGHT')))
#print("Goal test: " +  str(p1.goal_test(p1.result(state=(2, 2), action='RIGHT'))))



#BFS
start_time = time.time()
#time.sleep(1)
sol1 = breadth_search_graph(p1)
end_time = time.time()
show_solution("BFS su albero", sol1) 
print("Time = " + str("{0:.16f}".format(end_time - start_time)))
'''
# UCS
start_time = time.time()
time.sleep(1)
sol1 = bestFirst_search_graph(p1, lambda node: node.path_cost)
end_time = time.time()
show_solution("UCS", sol1)
print("Time = " + str("{0:.16f}".format(end_time - start_time)))

# greedy best first (manhattan)
start_time = time.time()
time.sleep(1)
sol2 = greedy_best_first_search2(p1)
end_time = time.time()
show_solution("greedy manhattan", sol2)
print("Time = " + str("{0:.16f}".format(end_time - start_time)))
'''
# greedy best first
start_time = time.time()
#time.sleep(1)
sol2 = greedy_best_first_search(p1)
end_time = time.time()
show_solution("greedy", sol2)
print("Time = " + str("{0:.16f}".format(end_time - start_time)))
'''
# A* (manhattan)
start_time = time.time()
time.sleep(1)
sol3 = astar_search2(p1)
end_time = time.time()
show_solution("A*", sol3)
print("Time = " + str("{0:.16f}".format(end_time - start_time)))
'''
# A*
start_time = time.time()
#time.sleep(1)
sol3 = astar_search(p1)
end_time = time.time()
show_solution("A*", sol3)
print("Time = " + str("{0:.16f}".format(end_time - start_time)))
