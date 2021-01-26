import cv2
import numpy as np
from time import gmtime, strftime
from os import listdir
from os.path import isfile, join


class AR_Gallery():
    def __init__(self, img):
        self.img = img
        self.ht, self.wt, self.dt = self.img.shape

        # Usando ORB para realizar a detecção
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.img, None)

        self.detected = False

        self.h, self.w = 1080, 1920

        self.get_gallery()

        self.vid_path = 'gallery/videos/'

        self.start_video(self.vid_path+'paint-cam video demo.avi')

    #def show_match(self):
    #    return cv2.drawMatchesKnn(self.img,kp1,frame_rsz,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def get_gallery(self):
        self.imgs = [f for f in listdir('gallery/imgs/') if isfile(join('gallery/imgs/', f))]
        self.videos = [f for f in listdir('gallery/videos/') if isfile(join('gallery/videos/', f))]

    def matches(self, des2):
        good = []
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1,des2,k=2)
        # Apply ratio test
        for m,n in matches:
            if (m.distance < 0.75*n.distance):
                good.append([m])
        return good
        

    def homography(self, frame, kp2, img_h, good):
        self.detected =True
                
        #homografia
        src_pts = np.float32([ self.kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

        #warp = cv2.warpPerspective(capa, M, (frame_rsz.shape[1], frame_rsz.shape[0]))

        pts = np.float32([ [0,0], [0,self.ht], [self.wt,self.ht], [self.wt,0] ]).reshape(-1,1,2)
        d = cv2.perspectiveTransform(pts, M)

        #MASCARA
        mask2 = np.ones((frame.shape[0], frame.shape[1], 3), np.uint8)*255
        cv2.fillPoly(mask2, [np.int32(d)], (0,0,0))
        
        frame_copy = frame.copy()
        img_ar = cv2.bitwise_and(frame_copy, mask2)
        img_warp = cv2.warpPerspective(img_h, M, (frame.shape[1], frame.shape[0]))
        #cv2.imshow('img_warp', img_warp)
        frame_rsz = cv2.bitwise_or(img_warp, img_ar)

        return frame_rsz



    def start_video(self, vid):
        self.type = 'vid'
        self.video_ar = cv2.VideoCapture(vid)
        sucess, imgVideo = self.video_ar.read()
        self.imgVideo = cv2.resize(imgVideo, (self.wt, self.ht))
        self.framecounter = 0

    def stop_video(self):
        if self.type == 'vid':
            self.video_ar.release()

    def run_app(self, frame):
        
        #operações para rodar video
        if self.detected == False:
            self.video_ar.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.framecounter = 0
        else:
            #caso acabe o video, repete
            if self.framecounter == self.video_ar.get(cv2.CAP_PROP_FRAME_COUNT):
                self.video_ar.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.framecounter = 0
            sucess, imgVideo = self.video_ar.read()
            self.imgVideo = cv2.resize(imgVideo, (self.wt,self.ht))
        
        frame_rsz = cv2.resize(frame, (self.w//2, self.h//2)) 

        # pegando os descritores dos frames
        kp2, des2 = self.orb.detectAndCompute(frame_rsz, None)

        # MATCH
        try:
            good = self.matches(des2)
            if( len(good) > 20 ):
                frame_rsz = self.homography(frame_rsz, kp2, self.imgVideo , good)
        except:
            pass
        
        self.framecounter += 1
        return frame_rsz

        
