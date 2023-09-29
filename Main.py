import PySimpleGUI as sg
import os
import cv2
from image_analysis import image_resize, scan_image, check_matrix
from Labyrinth import Labyrinth
from EasyMaze import EasyMaze
from solver import show_result



def start():
    file_types = [
        ("JPEG (*.jpg)", "*.jpg"),
        ("All files (*.*)", "*.*")
    ]

    # first column, where we can choose and show the image from the computer
    choose_file_column = [
        [sg.Image(key="-IMAGE-")],
        [
        sg.Input(size=(25, 1), enable_events=True, key="-FILE-"),
        sg.FileBrowse(file_types=file_types, key="-BROWSE-"),
        sg.Button("Load Image", key="-LOAD-")
        ]
    ]

    # general layout of the window
    layout = [
        [sg.Column(choose_file_column, key="-CHOOSE_COLUMN-"),
        sg.VSeparator(),
        sg.Column([], key="-CHECK_COLUMN-")]
    ]

    window = sg.Window("Image Viewer", layout, location=(300, 200))

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            exit()
            
        if event == '-LOAD-':
            filename = values["-FILE-"]
            if os.path.exists(filename):
                img = cv2.imread(filename)
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                original_shape = img.shape # save original dimensions to reconvert the image to these before the classification of the cells
                img = image_resize(img, 400, 400) # resize image to display it in the gui window
                imgbytes = cv2.imencode(".png", img)[1].tobytes() # necessary to display the image in the Image element of PySimpleGui
                window["-IMAGE-"].update(data=imgbytes)
                window["-FILE-"].update(visible=False)
                window["-BROWSE-"].update(visible=False)
                window["-LOAD-"].update(visible=False)
                # add 'Continue' and 'Choose a new image'  buttons to the first column
                window.extend_layout(window["-CHOOSE_COLUMN-"], [[sg.Button('Choose a new image', border_width=0, key='-RESTART-'),
                                                                    sg.Push(), # align the second button to the right
                                                                    sg.Button('Continue', border_width=0, key='-CONTINUE-')]])
        if event == '-CONTINUE-':
            img = image_resize(img, original_shape[0],  original_shape[1])
            matrix = scan_image(img)
            window["-CONTINUE-"].update(visible=False)
            matrix = check_matrix(matrix, window) # return None if we pressed again the button 'Choose a new image', after the analysis of the matrix
            if matrix == None:
                start()
            else:
                break
        
        if event == '-RESTART-':
            window.close()
            start()

    l = Labyrinth(matrix)
    p1 = EasyMaze(l)
    show_result(p1, l)



start()