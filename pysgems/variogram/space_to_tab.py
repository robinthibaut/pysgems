import numpy as np

def main(inputfile):
    file = open(inputfile, 'r')
    lines = file.readlines()
    list = []
    for line in lines:
        parts = line.split(sep=' ')
        #print(parts)
        newline = parts[0] + '\t' + parts[1] + '\t' + parts[2] +  '\t' + parts[3] + '\n' #+' ' + parts[4] + ' ' + parts[5] + ' ' + parts[6]
        with open(inputfile + 'spacetotabs' +'.txt', 'a') as the_file:
            the_file.write(newline)

    inputfilename = inputfile + 'spacetotabs'
    inputfile2 = inputfilename +'.txt'
    for line in open(inputfile2):
        if len(line) > 1 or line != '\n':
            #print(line, end='')
            with open(inputfilename + "no_whiterules.txt", 'a') as file:
                file.write(line)

