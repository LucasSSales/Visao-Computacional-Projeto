import cv2
import numpy as np
from time import gmtime, strftime
import imutils  

def put_png(png, bg):
    # pegando alpha do png
    alpha = png[:,:,3]
    # Criando mascara para remover a area do png
    _, mask_inv = cv2.threshold(alpha, 5, 255, cv2.THRESH_BINARY_INV)
    # conbertendo a mascara em rgba pra ter 4 canais
    mask_4depths = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGBA)
    # aplicvando no background 
    masked_bg = cv2.bitwise_and(bg, mask_4depths)
    # retorna o OR entre o png e fundo com mascara
    return cv2.bitwise_or(png, masked_bg)

# Dimensões da tela do meu celular
h_vid, w_vid = 1080, 1920

# Configurando o video da camera do meu celular
video = cv2.VideoCapture(0)
cam_ip = "https://10.0.0.104:8080/video"
video.open(cam_ip)

# Cascades para detecção de rosto, olhos e nariz
# Rosto do proprio repo do OpenCV, os outros são de terceiros
# Eyes Cascade (and others): https://kirr.co/694cu1
# Nose Cascade / Mustache Post: https://kirr.co/69c1le
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')


#pngs
glasses = cv2.imread("glasses.png", -1)
mustache = cv2.imread("mustache.png", -1)
hat = cv2.imread("Hat.png", -1)

while(True):
    ret, frame = video.read()
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        #Region Of Interest
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

        hat2 = imutils.resize(hat.copy(), width=w+200)
        #print("hat", hat.shape)
        #print("y", y)
        #print("hat2", hat2.shape)
        #cv2.imshow("hat", hat2)

        hat_h, hat_w, hat_d = hat2.shape
        start = 0
        down_hat = 100
        hat_area = frame[:y+down_hat, x-100:x+w+100]
        if ((y - hat_h)+down_hat < 0):
            #print("caiu no 1")
            start = hat_h - (y +down_hat)
            #print(hat2[start:,:].shape)
            #print(hat_area.shape)
            frame[:y+down_hat, x-100:x+w+100] = put_png(hat2[start:,:], hat_area)
           
        else:
            print("caiu no 2")
            hat_area = frame[y+down_hat-hat_h:y+down_hat, x-100:x+w+100]
            #print(hat2.shape)
            #print(hat_area.shape)
            frame[y+down_hat-hat_h:y+down_hat, x-100:x+w+100] = put_png(hat2, hat_area)
            

        #cv2.imshow("hat", hat2[start:,:])

        

        #cv2.imshow("tela onde vo por chapeu", hat_area)

        

        #print(hat2.shape, y)

        # DETECÇÃO DOS OLHOS NA AREA DO ROSTO
        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)

        for (ex,ey,ew,eh) in eyes:
            roi_eyes = roi_gray[ey:ey+eh, ex:ex+eh]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)

            glasses_zone = roi_color[ey-7:ey+eh+7,ex-15:ex+ew+15]
            #glasses2 = image_resize(glasses.copy(), width=ew)
            gw, gh, gc = glasses_zone.shape
            glasses2 = cv2.resize(glasses.copy(), (gh, gw))
            
            #cv2.imshow('glasses2', glasses2)
            roi_color[ey-7:ey+eh+7,ex-15:ex+ew+15] = put_png(glasses2, glasses_zone)
            '''
            

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    #print(glasses[i, j]) #RGBA
                    if glasses2[i, j][3] != 0: # alpha 0
                        roi_color[ey + i, ex + j] = glasses2[i, j]'''

        # DETECÇÃO DE NARIZ NA AREA DO ROSTO
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=7)

        for (nx,ny,nw,nh) in noses:
            roi_noses = roi_gray[ny:ny+nh, nx:nx+nh]
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)

            start_line = ny+(nh//2)
            mustache_zone = roi_color[start_line-7:ny+nh+7,nx-15:nx+nw+15]
            mw, mh, mc = mustache_zone.shape
            mustache2 = cv2.resize(mustache.copy(), (mh, mw))

            roi_color[start_line-7:ny+nh+7,nx-15:nx+nw+15] = put_png(mustache2, mustache_zone)





    frame_rsz = cv2.resize(frame, (w_vid//2, h_vid//2))
    #glasses_rsz = cv2.resize(glasses, (w_vid, h_vid))
    #print(glasses.shape)
    cv2.imshow('Paint Webcam', frame_rsz)
    #cv2.imshow('glasses', glasses)
    

    # Comando de Teclado
    k = cv2.waitKey(1)

    # Para sair, aperte q, a imagem sera salva qnd fechar
    if  k == ord('q'):
        break

    # para salvar a imagem, aperte s
    if  k == ord('s'):
        output_name = 'screenshot_' + strftime("%Y%m%d_%H%M%S", gmtime()) +".jpg"
        cv2.imwrite(output_name, frame_rsz)






video.release()
cv2.destroyAllWindows()
