import os
for image in os.listdir(r"/home/slyzen/Documents/Originals"):
    os.system('convert ' + image + ' ' + image + '.txt')

