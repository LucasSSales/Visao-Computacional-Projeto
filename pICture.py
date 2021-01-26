import cv2
import numpy as np
from time import gmtime, strftime
#from testes.teste_modulos import funcao
from paint_cam.paint_cam import Paint_cam
from custom_photo.cascades import Custom_photo
from ar_gallery.ar_scenes import AR_Gallery

def start_rec(shape_vid):
    output_name = videos_folder + 'output_' + strftime("%Y%m%d_%H%M%S", gmtime()) +".avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_name, fourcc, 15.0, shape_vid)


def stop_rec(out):
    out.release()

is_recording = False

imgs_folder = "gallery/imgs/"
videos_folder = "gallery/videos/"


# Shape da camera (1080, 1920, 3)
shape_vid = (540, 960)
#shape_vid = (1343, 568) # com a imagem da capa

video = cv2.VideoCapture(0)
cam_ip = "https://10.0.0.104:8080/video"
#cam_ip = "https://192.168.0.3:8080/video"
video.open(cam_ip)


flip = False

h, w = 1080, 1920

p = Paint_cam()
c = Custom_photo()
ar = AR_Gallery(cv2.cvtColor(cv2.imread('capa.jpg'), cv2.COLOR_BGR2RGB))

mode = p

face_cascade = cv2.CascadeClassifier('custom_photo/data/haarcascade_frontalface_default.xml')

def x(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    fr = cv2.cvtColor(f, cv2.COLOR_BGR2BGRA)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)


mode_str = 'paint_cam'


while(True):
    ret, frame = video.read()

    if flip:
        frame = cv2.flip(frame, 1)

    try:
        if mode is not None:
            frame = mode.run_app(frame)
    except:
        pass

    cv2.putText(frame, "Modo: "+mode_str, (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    frame_rsz = cv2.resize(frame, (w//2, h//2))

    rec_frame = frame_rsz.copy()

    
    if is_recording:
        frame_rsz = cv2.rectangle(frame_rsz, (840,20), (895,65), (0,0,255), -1)
        cv2.putText(frame_rsz, "REC", (855, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(rec_frame)

    
    cv2.imshow('pICture', frame_rsz)

    k = cv2.waitKey(1)

    if mode_str == 'paint_cam':
        mode.comandos(k)

    if  k == ord('r'):
        if is_recording :
            is_recording = False
            out.release()
        else:
            is_recording = True
            out = start_rec((w//2, h//2))

    # para salvar a imagem, aperte s
    if  k == ord('s'):
        output_name = imgs_folder + 'paint_' + strftime("%Y%m%d_%H%M%S", gmtime()) +".jpg"
        cv2.imwrite(output_name, frame)

    # Para sair, aperte q, a imagem sera salva qnd fechar
    if  k == ord('q'):
        break

    #flipa a tela
    if  k == ord('f'):
        flip = not flip

    # alterna entre os modos
    if  k == ord('1'):
        mode_str = 'paint_cam'
        mode = p

    if  k == ord('2'):
        mode_str = 'custom_photo'
        mode = c

    if  k == ord('3'):
        mode_str = 'ar_gallery'
        mode = ar

    #if  k == ord('4'):
    #    mode_str = 'rec'
    #    mode = None



video.release()
cv2.destroyAllWindows()