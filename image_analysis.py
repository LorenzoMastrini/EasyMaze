import math
import cv2
import numpy as np
import tensorflow as tf
import PySimpleGUI as sg
from matplotlib import pyplot as plt
from cv2.ximgproc import thinning



# resizes images maintaining aspect ratio
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized



# used by my_sort_contours() funcition to flatten a list of lists
from itertools import chain
def flatten_chain(list_of_lists):
	return list(chain.from_iterable(list_of_lists))



#scan the matrix to read each cell from left to right and from top to bottom
def sort_contours(cnts):
	# construct the list of bounding boxes
	boundingBoxes_og = [list(cv2.boundingRect(c)) for c in cnts] # original list
	boundingBoxes = list(boundingBoxes_og) # copy of the list eventually modified
	
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][1], reverse=False)) # sort the list of cells from top to bottom
	pairs_list = [] # this will contain alle the final (contours, boundingBoxes) pairs for each cell of the matrix

	n_elem_per_row = int(math.sqrt(len(boundingBoxes))) # dimensions of the matrix
	# sort each line of the matrix from left to right
	for i in range(n_elem_per_row):
		first_elem_each_line = int(n_elem_per_row * i)
		pairs_list.append(zip(*sorted(zip(cnts[first_elem_each_line:first_elem_each_line+n_elem_per_row], boundingBoxes[first_elem_each_line:first_elem_each_line+n_elem_per_row]), key=lambda b:b[1][0], reverse=False)))
	
	# unzip the result
	(cnts, boundingBoxes) = zip(*pairs_list)
	
	# flatten the results that otherwise would have been a list of lists (one list per line of the matrix)
	cnts = flatten_chain(cnts)
	boundingBoxes = flatten_chain(boundingBoxes)

	# return the final list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)



def scan_image(img):
	#lista_immagini = []

	img_width = np.array(img).shape[1]
	img_height = np.array(img).shape[0]
	
	# Thresholding the image
	(thresh, img_bin) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# Invert the image
	img_bin = 255 - img_bin
	
	'''
	# for debugging
	cv2.imshow("Image_bin", img_bin)
	cv2.waitKey(0)
	cv2.destroyWindow("Image_bin")
    '''

	# Defining a kernel length
	kernel_length = img_width // 30
	kernel_height = img_height // 30

	# A verticle kernel of (1 X kernel_length), which will detect all the vertical lines from the image
	verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))

	# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image
	hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

	# A kernel of (3 X 3) ones
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

	# Morphological operation to detect vertical lines from an image
	img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
	verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

	# Morphological operation to detect horizontal lines from an image
	img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
	horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
	
	'''
	# for debugging
	cv2.imshow("verticle_lines", verticle_lines_img)
	cv2.imshow("horizontal_lines", horizontal_lines_img)
	cv2.waitKey()
	cv2.destroyWindow("verticle_lines")
	cv2.destroyWindow("horizontal_lines")
	'''

	# Weighting parameters, this will decide the quantity of an image to be added to make a new image
	alpha = 0.5
	beta = 1.0 - alpha

	# This function helps to add two image with specific weight parameter to get a third image as summation of two image
	img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
	img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
	(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#cv2.imshow("img_final_bin", img_final_bin)
	#cv2.waitKey()
	#cv2.destroyAllWindows()

	# crop the image to keep only the matrix (removing the rest of the white page)
	contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	idx = 1
	mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
	cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
	out = np.zeros_like(img) # Extract out the object and place into output image
	out[mask == 255] = img[mask == 255]

	# Now crop
	(y, x) = np.where(mask == 255)
	(topy, topx) = (np.min(y), np.min(x))
	(bottomy, bottomx) = (np.max(y), np.max(x))
	img_final_bin = img_final_bin[topy:bottomy+1, topx:bottomx+1]
	img = img[topy:bottomy+1, topx:bottomx+1]
	img_bin = img_bin[topy:bottomy+1, topx:bottomx+1]

	'''
	# for debugging
	cv2.imshow('Cropped image', img_final_bin)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
    '''

	# Find contours for image, which will detect all the boxes
	contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	threshold_area = 1000 # threshold area
	contours_copy = []
	for cnt in contours:
		area = cv2.contourArea(cnt)         
		if area > threshold_area and area < 40000:
			contours_copy.append(cnt)
	n_elements = len(contours_copy)
	n_elements_per_row = int(math.sqrt(n_elements))

	'''
	# for debugging
	img_copy = cv2.drawContours(img, contours_copy, -1, (0, 0, 255), 3)
	img_copy = image_resize(img, height=600)
	cv2.imshow("img_contours", img_copy)
	cv2.waitKey()
	cv2.destroyWindow("img_contours")
	'''

	# Sort all the contours from top to bottom
	(contours, boundingBoxes) = sort_contours(contours_copy)

	model = tf.keras.models.load_model('handwritten_recognition.model')

	class_labels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 'E', 6: 'S', 7: 'X'} # values associated to each label

	matrix = []
	for i in range(n_elements_per_row): # create a list for each of the row that the labyrinth will have
		matrix.append([])

	idx = 0 # cell contour index
	line = 0 # line counter
	cc = 0 # counter of the elements per line (it goes always from 0 to 6, for each line)
	for c in contours:
		# Returns the location and width,height for every contour
		x, y, w, h = cv2.boundingRect(c)

		# If the box height and widht are >10 and < 1000 (filter out unwanted contours), read the image in it to classify the digit inside
		if (w > 10 and h > 10 and w < 1000 and h < 1000):
			new_img = img_bin[y:y+h, x:x+w]
			new_img = cv2.resize(new_img, (28, 28)) # we need 28x28 images to classify the digits with the neural network
			
			#clean the borders of the image to delete informations non related to the text
			new_img[0:4, 0:28] = 0
			new_img[0:28, 0:4] = 0
			new_img[28-4:28, 0:28] = 0
			new_img[0:28, 28-4: 28] = 0
			
			thin = thinning(new_img, cv2.ximgproc.THINNING_ZHANGSUEN) # find the skeleton of the numer/letter
			thin = cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)), iterations=1) # make the skeleton a bit larger
			
			# Find the contours on the image
			contours, hierarchy = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for _, c in enumerate(contours):
				boundRect = cv2.boundingRect(c) # calculate the bounding box around the character

				# Get the bounding rectangle data
				rectX = boundRect[0]
				rectY = boundRect[1]
				rectWidth = boundRect[2]
				rectHeight = boundRect[3]

				# Estimate the bounding rect area:
				rectArea = rectWidth * rectHeight

				# Set a min area threshold
				minArea = 50

				# Filter blobs by area for two reasons:
				# 	- we avoid potential noise;
				# 	- we don't want to scale the 1s because otherwise they scale horizontally and destroy the shape of the digit, making it non recognisable
				if rectArea > minArea:
					# Crop bounding box:
					currentCrop = thin[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
					
					'''
					# for debugging
					cv2.imshow("Current Crop", currentCrop)
					cv2.waitKey(0)
					'''

					# scale the image to a dimension of 23 pixel in one of the two dimensions, mantaining the aspect ratio
					if (currentCrop.shape[1] >= currentCrop.shape[0]):
						pixels_to_zoom = 23 - currentCrop.shape[1]
						currentCrop = cv2.resize(currentCrop, (currentCrop.shape[1] + pixels_to_zoom, currentCrop.shape[0] + pixels_to_zoom), interpolation=cv2.INTER_AREA)
					else:
						pixels_to_zoom = 23 - currentCrop.shape[0]
						currentCrop = cv2.resize(currentCrop, (currentCrop.shape[1] + pixels_to_zoom, currentCrop.shape[0] + pixels_to_zoom), interpolation=cv2.INTER_AREA)

					# calculate the amount of padding to reach the size of 28x28 in the horizontal and vertical dimensions
					if (currentCrop.shape[1] < 28):
						padding_x = int((28 - currentCrop.shape[1]) / 2)
					if (currentCrop.shape[0] < 28):
						padding_y = int((28 - currentCrop.shape[0]) / 2)

					# actual padding of the image
					thin = cv2.copyMakeBorder(currentCrop, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
					
					# because of the roundings in the calculation of the padding the image miight not be exactly 28x28, so we need to manually add some black pixels
					flag = False # used to create padding to the borders alternatively between left and right and between top and bottom, so that the digit stays in the "middle" of the image
					if thin.shape[1] % 2 != 0:
						if flag:
							thin = cv2.copyMakeBorder(thin, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
							flag = False
						else:
							thin = cv2.copyMakeBorder(thin, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
							flag = True
					if thin.shape[0] % 2 != 0:
						if flag:
							thin = cv2.copyMakeBorder(thin, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
							flag = False
						else:
							thin = cv2.copyMakeBorder(thin, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
							flag = True
			
			'''
			# for debugging
			cv2.imshow("Cell no." + str(idx + 1), new_img)
			cv2.waitKey(0)
			'''

			x = np.array([thin]) # necessary step to predict the content of the matrix
			y = x.reshape((28, 28))
			x = x / 255.0

			#lista_immagini.append(y)

			prediction = model.predict(x)
			predicted_label = class_labels.get((np.argmax(prediction)))
			print(f"The digit is probably a {predicted_label}")

			matrix[line].append(predicted_label)
			cc += 1

			if (cc == n_elements_per_row): # we reached the end of the line, so we go to the next one
				cc = 0
				line += 1

			if cv2.waitKey(1) & 0xFF == ord('q'): # waitKey(0) for debugging with imshow()
				cv2.destroyAllWindows()
				exit()
		 
		idx += 1

	'''
	# for debugging
	plt.figure(figsize=(10,10))
	for i in range(len(lista_immagini)):
		plt.subplot(int(math.sqrt(len(lista_immagini))), int(math.sqrt(len(lista_immagini))), i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(lista_immagini[i], cmap=plt.cm.binary)
	plt.show()
	'''
	
	return matrix



def check_matrix(mat, window):
	window.extend_layout(window["-CHECK_COLUMN-"], [[sg.Text('Is the matrix correct?', font=('Calibri', 14)), sg.Button('Yes', border_width=0, key='-YES-', font=('Calibri', 12))]])
	layout_buttons = [] # list that will save all the buttons that represent each cell of the matrix
	
	# fit all the buttons with the predicted labels
	for row in range(len(mat)):
		new_row = []
		for column in range(len(mat)):
			new_row.append(sg.Button(mat[row][column], size=(3, 1), button_color=('black', 'white'), key=(row,column), border_width=2, pad=0, font=('Calibri', 14)))
		layout_buttons.append(new_row)
	
	window.extend_layout(window["-CHECK_COLUMN-"], layout_buttons)

	while True:
		event, values = window.read()
		
		if event == sg.WIN_CLOSED:
			exit()

		if event == '-YES-':
			break
		
		# continuously check if we press the button corresponding to a cell to update the value (if it has been misclassified)
		for row in range(len(mat)):
			for column in range(len(mat)):
				if event == (row, column): # detect one specific cell based on its coordinates
					text = sg.popup_get_text('Enter the correct value')
					if (text == 'e' or text == 's' or text == 'x'): # we only accept capitalized letters
						text = text.capitalize()
					elif str(text).isdigit(): # if the text is a number we convert it to int
						text = int(text)
					
					if text != '' and text != None:
						mat[row][column] = text
						window[(row,column)].update(text=text)

		if event == '-RESTART-':
			return window.close()
			
	window.close()
	return mat
