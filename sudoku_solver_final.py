#!/usr/bin/env python3
"""
COMP4102    final project
=================================
authors:    Wenyu Zhang, Bowen Zuo
title:      sudoku solver
date:       April 18, 2020
=================================
"""

import cv2
import operator
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR/tesseract'
#img  =  cv2.imread('sudoku.jpg')

"""
show some images
"""
def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    

"""
locate the sudoku corners
"""
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):

    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_not(img)
    
    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    show_image("Corner Locations", img)
    return img


"""
img process
"""
def preprocess_img(img):

    preprocess  = cv2.GaussianBlur(img.copy(), (7, 7), 0)
    preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocess = cv2.bitwise_not(preprocess, preprocess)
    return preprocess

"""
find counters
"""
def find_contours(processed_image):

    ext_contours, hier = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    #show_image("", processed_image)
    external_contours = cv2.drawContours(processed_image.copy(), ext_contours, -1, (255, 0 ,0 ), 2)

	
"""
locate 4 corners of sudoku
"""
def get_corners_of_largest_poly(img):

    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True) #Sort by area, descending

    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


"""
crop sudoku

"""
def infer_sudoku_puzzle(image, crop_rectangle):

    img = image
    crop_rect = crop_rectangle
	
    def distance_between(a, b): 
        return np.sqrt( ((b[0] - a[0]) **2) + ((b[1] - a[1]) **2) )

    def crop_img(): 
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
        source_rect = np.array(np.array([top_left, bottom_left, bottom_right, top_right], dtype='float32')) #float for perspective transformation
        side = max([distance_between(bottom_right, top_right), 
		    distance_between(top_left, bottom_left),
		    distance_between(bottom_right, bottom_left),
		    distance_between(top_left, top_right)
		    ])
        dest_square = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
        m = cv2.getPerspectiveTransform(source_rect, dest_square)
        return cv2.warpPerspective(img, m, (int(side), int(side)))

    return crop_img()




"""
scan each cell
save number into array

"""
def extract_digit(img): 
    digits = np.zeros((9,9),dtype=np.int8)
    board = ''
    side = img.shape[:1]
    side = side[0] / 9
    margin = side / 12
    print("Extracting Digit.....")
    for i in range(9):
        for j in range(9):
            temp = img[j*side+margin:(j+1)*side-margin, i*side+margin:(i+1)*side-margin]
            _,temp = cv2.threshold(temp, 190, 255, cv2.THRESH_BINARY)
            #temp = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            #cv2.imwrite(('tmp/'+str(i)+str(j)+'.png'), temp)

            text = pytesseract.image_to_string(temp,config='--psm 10 digits')
            text = [ int(letter) for letter in text if letter.isdigit() ]
            if len(text)==0:
                text = 0
            else:
                #print (text)
                #print (" ")
                text = text[0]
            #print(text)
            board += str(text)
            digits[j,i] = text
        #print('')
        board += '\n'
    return digits




# =========================== solving functions =========================== #
"""
fucntions for solving the sudoku
"""

#helpers
def get_row(puzzle, row_num):
    return puzzle[row_num]

def get_column(puzzle, col_num):
    return [puzzle[i][col_num] for i, _ in enumerate(puzzle[0])]

def get_square(puzzle, row_num, col_num):
    square_x = row_num // 3
    square_y = col_num // 3
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((square_x * 3 + j, square_y * 3 + i))
    return [puzzle[i[0]][i[1]] for i in coords]

def get_possibilities(puzzle, row_num, col_num):
    possible = set(range(1, 10))
    row = get_row(puzzle, row_num)
    col = get_column(puzzle, col_num)
    square = get_square(puzzle, row_num, col_num)
    not_possible = set(row + col + square)
    return possible - not_possible


def check_if_solvable(unsolved_puzzle):
    for i in range(9):
        if sum(set(get_row(unsolved_puzzle, i))) != sum(get_row(unsolved_puzzle, i)):
            return False
        if sum(set(get_column(unsolved_puzzle, i))) != sum(get_column(unsolved_puzzle, i)):
            return False
        if sum(set(get_square(unsolved_puzzle, i, i))) != sum(get_square(unsolved_puzzle, i, i)):
            return False
    return True


def solve(puzzle):
    solved = True
    
    for row, row_values in enumerate(puzzle):
        for col, value in enumerate(row_values):
            if value == 0:
                solved = False
                break
        else:
            continue
        break
    if solved:
        return puzzle
    for i in range(1, 10):
        if i in get_possibilities(puzzle, row, col):
            puzzle[row][col] = i
            if solve(puzzle):
                return puzzle
            else:
                puzzle[row][col] = 0
    return False


def verify(solved_puzzle):
    for i in range(9):
        if sum(get_row(solved_puzzle, i)) != 45:
            return False
        if sum(get_column(solved_puzzle, i)) != 45:
            return False
        if sum(get_square(solved_puzzle, i, i)) != 45:
            return False
    return True



# ==================  end of solving function  ===========================#

"""
print list output to formated text
"""
def printResult(out):
    print("\nFinal Output:\n=======================")
    for i in range(9):
        output=''
        for j in range(9):
            if j == 2 or j ==5 or j == 8:
                output = output+ "|"+str(out[i][j])+"| "
            else:
                output = output+  "|"+str(out[i][j])
        print(output)
        if i == 2 or i == 5:
            print('-----------------------')


"""
fill answer back to the cropped sudoku
"""
def fillBack(result, img, digit):

    side = img.shape[:1]
    side = side[0] / 9
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(9):
        for j in range(9):
            coords = tuple((i*side+10,(j+1)*side-10))
            text = str(result[j][i])
            if digit[j,i]==0:
                cv2.putText(img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    return img


"""
main fucntion
process image and solve the sudoku
"""
def main():
    img = cv2.imread('sudoku.jpg', cv2.IMREAD_GRAYSCALE)
    show_image("Original Sudoku", img)

    processed_sudoku = preprocess_img(img)

    find_contours(processed_sudoku)
    corners_of_sudoku = get_corners_of_largest_poly(processed_sudoku)
    display_points(img, corners_of_sudoku)
    cropped_sudoku = infer_sudoku_puzzle(img, corners_of_sudoku)
    img_out = cropped_sudoku.copy()
    show_image("Cropped Sudoku", cropped_sudoku)

    #cropped = infer_sudoku_puzzle(img, corners_of_sudoku)
    #cropped = cv2.bitwise_not(cropped)

    digit = extract_digit(cropped_sudoku)
    print(digit)
    print("Solving....")
    output = solve(digit.tolist())
    if not output:
        raise ValueError('ERROR: not solvable')
    printResult (output)
    fill = fillBack(output,img_out,digit)
    show_image("Final Output",fill)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
