import PySimpleGUI as sg
from EasyMaze import breadth_search_tree, solution, bestFirst_search_graph, greedy_best_first_search_Manhattan, greedy_best_first_search, astar_search_Manhattan
from Librerie.aima.search import astar_search



def bfs(problem):
    #BFS
    sol1 = breadth_search_tree(problem)
    sol_bfs = solution(sol1) 
    return sol_bfs



def ucs(problem):
    # UCS
    sol1 = bestFirst_search_graph(problem, lambda node: node.path_cost)
    sol_ucs = solution(sol1)
    return sol_ucs



def greedy_best_first(problem, manhattan=False):
    if (manhattan):
        # greedy best first (manhattan)
        sol2 = greedy_best_first_search_Manhattan(problem)
        sol_greedy = solution(sol2)
    else:
        # greedy best first
        #time.sleep(1)
        sol2 = greedy_best_first_search(problem)
        sol_greedy = solution(sol2)
    return sol_greedy



def a_star(problem, manhattan=False):
    if(manhattan):
        # A* (manhattan)
        sol3 = astar_search_Manhattan(problem)
        sol_astar = solution(sol3)
    else:
        # A*
        sol3 = astar_search(problem)
        sol_astar = solution(sol3)
    return sol_astar



def show_result(p1, labyrinth):
    solutions_list = []
    solutions_list.append(bfs(p1))
    solutions_list.append(greedy_best_first(p1))
    solutions_list.append(a_star(p1))

    # layout used to represent the matrix
    layout_buttons = []

    # general layout of the window
    layout = [[sg.Column(layout_buttons, key="-FINAL_COLUMN-")]]

    # layout showed if there is a solution for the problem
    layout_solution = [[
            sg.Button('BFS', key='-BFS-', border_width=2, auto_size_button=True, pad=7),
            sg.Button('Greedy Best First', key='-GREEDY-', border_width=2, auto_size_button=True, pad=7),
            sg.Button('A*', key='-ASTAR-', border_width=2, auto_size_button=True, pad=7)
    ]]

    # layout showed if there isn't any solution
    layout_no_solution = [
        [sg.Text('No solution', font=('Calibri', 14), text_color='black', justification='center', border_width=2, pad=7)]
    ]
    
    
    # buttons initialization (each button is a cell of the matrix)
    for row in range(len(labyrinth.matrix)):
        new_row = [] # compile a row per cycle
        for column in range(len(labyrinth.matrix)):
            if (labyrinth.matrix[row][column] == 'X'):
                new_row.append(sg.Button(size=(3, 1), button_color=('black', 'black'), key=(row,column), border_width=2, pad=0, disabled=True, font=('Italic', 14))) # walls are black
            elif (labyrinth.matrix[row][column] == 'S'):
                new_row.append(sg.Button(labyrinth.matrix[row][column], size=(3, 1), button_color=('black', 'blue'), key=(row,column), border_width=2, pad=0,disabled=True, disabled_button_color=('black', 'blue'), font=('Italic', 14))) # initializing starting point to blue
            else:
                new_row.append(sg.Button(labyrinth.matrix[row][column], size=(3, 1), button_color=('black', 'white'), key=(row,column), border_width=2, pad=0, disabled=True, disabled_button_color=('black', 'white'), font=('Italic', 14)))
        layout_buttons.append(new_row)

    window = sg.Window(title="Show result", layout=(layout_buttons, layout), margins=(100, 50), finalize=False)

    cell = labyrinth.entrance # starting cell
    cell = list(cell) # necessary conversion
    solution = []
    action = 0 # variable that scan each cell in the solution

    flag = False # flag to indicate if the layout has been initialized (wheter there is a solution or not)
    while True:
        event, values = window.read(timeout=(1000))

        if event == sg.WIN_CLOSED:
            exit()
    
        if event == '-BFS-':
            solution = solutions_list[0]
        elif event == '-GREEDY-':
            solution = solutions_list[1]
        elif event == '-ASTAR-':
            solution = solutions_list[2]

        if flag == False: # layout has not been initialized so we do it (only once, at the beginning of the while cycle)
            if len(solutions_list[0]) == 0: # no solution
                window.extend_layout(window["-FINAL_COLUMN-"], layout_no_solution)
            else:
                window.extend_layout(window["-FINAL_COLUMN-"], layout_solution)
            flag = True
        
        if solution: # if solution is not empty
            window[(cell[0], cell[1])].update(button_color=('black', 'white'))
            if (action < len(solution)):
                if solution[action] == 'UP':
                    cell[0] = cell[0] - 1
                    window[(cell[0], cell[1])].update(button_color=('black', 'blue')) # previous row, same column
                elif solution[action] == 'RIGHT':
                    cell[1] = cell[1] + 1
                    window[(cell[0], cell[1])].update(button_color=('black', 'blue')) # previous row, same column
                elif solution[action] == 'DOWN':
                    cell[0] = cell[0] + 1
                    window[(cell[0], cell[1])].update(button_color=('black', 'blue')) # previous row, same column
                elif solution[action] == 'LEFT':
                    cell[1] = cell[1] - 1
                    window[(cell[0], cell[1])].update(button_color=('black', 'blue')) # previous row, same column            
        
                action += 1
            else: # all solution has been printed
                # re-initialize everything for the next cycle
                cell = labyrinth.entrance
                cell = list(cell) # necessary conversion
                window[(cell[0], cell[1])].update(button_color=('black', 'blue'))
                solution = []
                action = 0
        
            
        
                