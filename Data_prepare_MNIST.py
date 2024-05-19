import os
import numpy
import cv2
import numpy as np

nr_train_samples = 30
nr_test_samples = 10

liste = []
row = 1
while row < 29:
    column = 1
    string = ""
    while  column < 29:
        liste.append("R" + str(row) + "C" + str(column))
        column += 1
    row += 1

train_file = open("Fashion_train.csv", "w")
test_file = open("Fashion_test.csv", "w")

string1 = ""
string2 = ""
for elem in liste:
    string1 += elem + ","
    string2 += elem + ","
string1 += "path,label\n"
string2 += "path,label\n"
train_file.write(string1)
test_file.write(string2)

cwd = os.getcwd()

train_path = cwd + "/fashion_mnist_images/train/"
test_path = cwd + "/fashion_mnist_images/test/"

def adjust_num(num):
    base = ['0','0','0','0']
    num_str = str(num)[::-1]
    #num_str = num_str
    b = 0
    while b < len(num_str):
        base[b] = num_str[b]
        b += 1
    txt = "".join(base)
    return txt[::-1]

def add_pic_matrix(matrix_path , file , label, last=False):
    image_data = cv2.imread(matrix_path , cv2.IMREAD_UNCHANGED)
    appender = ""
    for row in image_data:
        for elem in row:
            appender += str(elem) + ","
    if last == False:
        appender += matrix_path + "," + str(label) + "\n"
    else:
        appender += matrix_path + "," + str(label)
    file.write(appender)

np.set_printoptions(linewidth=200)

i = 0
while i < 10:
    a = 0
    while a < nr_train_samples :
        iter_train_path = train_path + str(i) + "/" + adjust_num(a) + ".png"
        if a == nr_train_samples-1 and i == 9:
            add_pic_matrix(iter_train_path, train_file, i, True)
        else:
            add_pic_matrix(iter_train_path, train_file, i)
        a += 1
    a = 0
    while a < nr_test_samples:
        iter_test_path = test_path + str(i) + "/" + adjust_num(a) + ".png"
        if a == nr_test_samples-1 and i == 9:
            add_pic_matrix(iter_test_path , test_file, i, True)
        else:
            add_pic_matrix(iter_test_path, test_file, i)
        a += 1
    i += 1
