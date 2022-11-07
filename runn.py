import pygame, sys
from pygame.locals import *
import numpy as np
import cv2
from keras.models import load_model


WHITE = (255,255,255)
BLACK = (0,0,0)
Red = (255,0,0)
img_cnt = 1
REDECT = True

IMAGESAVE  =False
Model = load_model(r"C:\Users\LENOVO\PycharmProjects\pythonProject6HindwriteRec\bestmodel.h5")
LABELS = {0:'ZERO',1:'ONE',2:'TWO',3:"THREE",4:'FOUR',5:'FIVE',6:"SIX",7:"SEVEN",8:"EIGHT",9:'NINE'}

windosizeX  = 640
windosizeY  = 480

#inizalize pygame
pygame.init()

Display_sur = pygame.display.set_mode((windosizeX, windosizeY))
pygame.display.set_caption("Digit board")
iswriting  = False

num_xcord = []
num_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(Display_sur,WHITE, (xcord,ycord),4,0)
            num_ycord.append(ycord)
            num_xcord.append(xcord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)

            rec_min_x, rec_max_x = max(num_xcord[0]-5,0), min(windosizeX,num_xcord[-1]+5)
            rec_min_y, rec_max_y = max(num_ycord[0] - 5, 0), min(num_ycord[-1] + 5,windosizeX)

            num_xcord = []
            num_ycord = []

            img_arr = np.array(pygame.PixelArray(Display_sur))[rec_min_x:rec_max_x, rec_min_y:rec_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("iamge.jpeg")
                img_cnt +=1

            if REDECT:
                image = cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10),"constant", constant_values=0)
                image = cv2.resize(image,(28,28))/225

                label1 =LABELS[np.argmax(Model.predict(image.reshape(1,28,28,1)))]


                print(label1)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    Display_sur.fill(BLACK)

        pygame.display.update()
