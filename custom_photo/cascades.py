import cv2
import numpy as np
from time import gmtime, strftime
import imutils  

class Custom_photo():

    def __init__(self):
        # Dimensões da tela do meu celular
        self.h_vid, self.w_vid = 1080, 1920

        # Cascades para detecção de rosto, olhos e nariz
        # Rosto do proprio repo do OpenCV, os outros são de terceiros
        # Eyes Cascade (and others): https://kirr.co/694cu1
        # Nose Cascade / Mustache Post: https://kirr.co/69c1le
        self.face_cascade = cv2.CascadeClassifier('custom_photo/data/haarcascade_frontalface_default.xml')
        self.eyes_cascade = cv2.CascadeClassifier('custom_photo/third-party/frontalEyes35x16.xml')
        self.nose_cascade = cv2.CascadeClassifier('custom_photo/third-party/Nose18x15.xml')

        # colocar pra ele buscar td na pasta imgs
        #pngs
        self.glasses = cv2.imread("custom_photo/imgs/glasses.png", -1)
        self.mustache = cv2.imread("custom_photo/imgs/mustache.png", -1)
        self.hat = cv2.imread("custom_photo/imgs/hat.png", -1)



    def put_png(self, png, bg):
        # pegando alpha do png
        alpha = png[:,:,3]
        # Criando mascara para remover a area do png
        _, mask_inv = cv2.threshold(alpha, 20, 255, cv2.THRESH_BINARY_INV)
        # conbertendo a mascara em rgba pra ter 4 canais
        mask_4depths = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGBA)
        # aplicvando no background 
        masked_bg = cv2.bitwise_and(bg, mask_4depths)
        # retorna o OR entre o png e fundo com mascara
        return cv2.bitwise_or(png, masked_bg)


    def run_app(self, frame):       
        #face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)


        for (x,y,w,h) in faces:
            #Region Of Interest
            roi_gray = gray[y:y+h, x:x+h]
            roi_color = frame[y:y+h, x:x+h]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

            hat2 = imutils.resize(self.hat.copy(), width=w+200)

            hat_h, hat_w, hat_d = hat2.shape
            start = 0
            down_hat = 100
            hat_area = frame[:y+down_hat, x-100:x+w+100]
            if ((y - hat_h)+down_hat < 0):
                start = hat_h - (y +down_hat)
                frame[:y+down_hat, x-100:x+w+100] = self.put_png(hat2[start:,:], hat_area)
            
            else:
                hat_area = frame[y+down_hat-hat_h:y+down_hat, x-100:x+w+100]
                frame[y+down_hat-hat_h:y+down_hat, x-100:x+w+100] = self.put_png(hat2, hat_area)
                

            # DETECÇÃO DOS OLHOS NA AREA DO ROSTO
            eyes = self.eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)

            for (ex,ey,ew,eh) in eyes:
                roi_eyes = roi_gray[ey:ey+eh, ex:ex+eh]
                #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)

                glasses_zone = roi_color[ey-7:ey+eh+7,ex-15:ex+ew+15]
                #glasses2 = image_resize(glasses.copy(), width=ew)
                gw, gh, gc = glasses_zone.shape
                glasses2 = cv2.resize(self.glasses.copy(), (gh, gw))
                
                #cv2.imshow('glasses2', glasses2)
                roi_color[ey-7:ey+eh+7,ex-15:ex+ew+15] = self.put_png(glasses2, glasses_zone)

            # DETECÇÃO DE NARIZ NA AREA DO ROSTO
            noses = self.nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=7)

            for (nx,ny,nw,nh) in noses:
                roi_noses = roi_gray[ny:ny+nh, nx:nx+nh]
                #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)

                start_line = ny+(nh//2)
                mustache_zone = roi_color[start_line-7:ny+nh+7,nx-15:nx+nw+15]
                mw, mh, mc = mustache_zone.shape
                mustache2 = cv2.resize(self.mustache.copy(), (mh, mw))

                roi_color[start_line-7:ny+nh+7,nx-15:nx+nw+15] = self.put_png(mustache2, mustache_zone)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return cv2.resize(frame, (self.w_vid//2, self.h_vid//2))
        