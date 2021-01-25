import cv2
import numpy as np
from time import gmtime, strftime

# Carregando as imagens
capa = cv2.cvtColor(cv2.imread('capa.jpg'), cv2.COLOR_BGR2RGB) # Imagem a ser detectada
maki = cv2.cvtColor(cv2.imread('maki.jpg'), cv2.COLOR_BGR2RGB) # Imagem que será exibida
ht, wt, dt = capa.shape
maki_rsz = cv2.resize(maki, (wt, ht))


# Usando ORB para realizar a detecção
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(capa, None)

detected = False

# Shape da camera (1080, 1920, 3)
shape_vid = (540, 960)
#shape_vid = (1343, 568) # com a imagem da capa

video = cv2.VideoCapture(0)
cam_ip = "https://10.0.0.104:8080/video"
video.open(cam_ip)

video_ar = cv2.VideoCapture('teste.avi')
sucess, imgVideo = video_ar.read()
imgVideo = cv2.resize(imgVideo, (wt, ht))
framecounter = 0


# Objetos para salvar o video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#output_name = 'output_' + strftime("%Y%m%d_%H%M%S", gmtime()) +".avi"
#out = cv2.VideoWriter(output_name, fourcc, 15.0, shape_vid)

h, w = 1080, 1920

while(True):
    ret, frame = video.read()

    #operações para rodar video
    if detected == False:
        video_ar.set(cv2.CAP_PROP_POS_FRAMES, 0)
        framecounter = 0
    else:
        #caso acabe o video, repete
        if framecounter == video_ar.get(cv2.CAP_PROP_FRAME_COUNT):
            video_ar.set(cv2.CAP_PROP_POS_FRAMES, 0)
            framecounter = 0
        sucess, imgVideo = video_ar.read()
        imgVideo = cv2.resize(imgVideo, (wt,ht))

        #flip na img caso eu fique mt confuso
        #frame = cv2.flip(frame, 1)
        #print(frame.shape)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
    
    frame_hsv_rsz = cv2.resize(frame_hsv, (w//2, h//2)) 
    frame_rsz = cv2.resize(frame, (w//2, h//2)) 


    # pegando os descritores dos frames
    kp2, des2 = orb.detectAndCompute(frame_rsz, None)


    # MATCH
    try:
        good = []
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        for m,n in matches:
            if (m.distance < 0.75*n.distance):
                good.append([m])
    
        if( len(good) > 20 ):
            detected =True
            #print('DETECTADO')
            
            #homografia
            src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
            #warp = cv2.warpPerspective(capa, M, (frame_rsz.shape[1], frame_rsz.shape[0]))

            pts = np.float32([ [0,0], [0,ht], [wt,ht], [wt,0] ]).reshape(-1,1,2)
            d = cv2.perspectiveTransform(pts, M)

            #MASCARA
            mask2 = np.ones((frame_rsz.shape[0], frame_rsz.shape[1], 3), np.uint8)*255
            cv2.fillPoly(mask2, [np.int32(d)], (0,0,0))

            frame_copy = frame_rsz.copy()
            img_ar = cv2.bitwise_and(frame_copy, mask2)
            maki_warp = cv2.warpPerspective(imgVideo, M, (frame_rsz.shape[1], frame_rsz.shape[0]))
            frame_rsz = cv2.bitwise_or(maki_warp, img_ar)


            #linhas
            frame_rsz = cv2.polylines(frame_rsz.copy(), [np.int32(d)], True, (255,0,255), 3)

            (x,y),radius = cv2.minEnclosingCircle(d)
            center = (int(x),int(y))
            radius = int(radius)
            frame_rsz = cv2.circle(frame_rsz,center,radius,(0,255,0),2)

            #frame_rsz = cv.circle(frame_rsz,center,30,(0,255,0),-1)
            
    except:
        pass


    match = cv2.drawMatchesKnn(capa,kp1,frame_rsz,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Celular', frame_rsz)
    #cv2.imshow('Tela', tela)
    #cv2.imshow('Fusao', cv2.bitwise_or(tela, frame_rsz))
    
    #cv2.imshow('Mask', mask)
    #out.write(match)
    #out.write(frame_rsz)

    k = cv2.waitKey(1)

    if  k == ord('q'):
        break

    if k == ord('s'):
        cv2.imwrite("screnshot_test.jpg", frame_rsz)

    framecounter += 1
    


video.release()
video_ar.release()
#out.release()
cv2.destroyAllWindows()