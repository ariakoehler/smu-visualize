import cv2
import numpy as np
import os,glob,sys
from leaptest import SampleListener

sys.path.insert(0,os.path.join(os.getcwd(),'../lib/'))

import Leap


class Cam:
    def __init__(self,device,resolution):
        #Initialize camera <device> at <resolution>
        self.device = device
        self.w = resolution[0]
        self.h = resolution[1]
        self.camera = cv2.VideoCapture(device)
        self.camera.set(3,self.w)
        self.camera.set(4,self.h)

        self.canvas = np.zeros((self.h,int(self.h*(16.0/9.0)),3), np.uint8)

        self.t = 0

        self.listener = SampleListener()
        self.controller = Leap.Controller()
        self.controller.add_listener(self.listener)

    def getFrame(self):
        return self.camera.read()[1]

    def calibImage(self):
        theFrame = self.camera.read()[1]
        cv2.imshow('Captured frame',theFrame)
        cv2.waitKey(0)
        cv2.imwrite('calibration%d.png'%self.device,theFrame)

        self.calibrate(1)

    def calibrate(self,show=False):
        if show:
            path = './calibration%d.png'%self.device
        else:
            path = './src/calibration%d.png'%self.device

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w = 7
        h = 7
        board = np.zeros((w*h,3), np.float32)
        board[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)
        board = board.reshape(-1,1,3)

        if len(glob.glob(path)) == 0:
            print("calibration file %d does not exist!!!"%self.device)
            raise ValueError
        img = cv2.imread(path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h,w),None)

        if ret == True:
            objpoints = [board]

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints = [corners2]

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (h,w), corners2,ret)
            if show:
                cv2.imshow('img',img)
                cv2.waitKey(0)

            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        else:
            print('No chess board found')
            return -1
        return img

    def project(self,points):
        inp = np.array(points,dtype=np.float32)
        projected,_ = cv2.projectPoints(inp, self.rvecs[0], self.tvecs[0], self.mtx, self.dist)
        return [(i[0][0],i[0][1]) for i in projected]

    def rotateX(self,points,theta,origin=[0.0,0.0,0.0]):
        rotation = np.matrix([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
        ret = []
        for p in points:
            translated = p-np.asarray(origin)
            ret.append((rotation*translated.reshape([3,1])).reshape([1,3])[0] + np.asarray(origin))
        return ret

    def rotateY(self,points,theta,origin=[0.0,0.0,0.0]):
        rotation = np.matrix([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
        ret = []
        for p in points:
            translated = p-np.asarray(origin)
            ret.append((rotation*translated.reshape([3,1])).reshape([1,3])[0] + np.asarray(origin))
        return ret

    def rotateZ(self,points,theta,origin=[0.0,0.0,0.0]):
        rotation = np.matrix([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
        ret = []
        for p in points:
            translated = p-np.asarray(origin)
            ret.append((rotation*translated.reshape([3,1])).reshape([1,3])[0] + np.asarray(origin))
        return ret


    def newBasis(self,vertices,basis,origin):
        ret = []
        basis = np.matrix(basis).transpose()
        origin = np.asarray(origin).reshape([3,1])
        for i in range(len(vertices)):
            scale = float(np.dot(np.asarray(vertices[i]), basis[:,i]))/float(np.dot(basis[:,i].transpose(), basis[:,i]))
            ret.append(scale * basis[:,i])
        return np.array(ret)

    def drawLine(self,surface,vertices,color,thickness=1,closed=0):
        transformed = self.project(vertices)
        for vertex in range(len(transformed)-1):
            cv2.line(surface,transformed[vertex],transformed[vertex+1],color,thickness)
        if closed:
            cv2.line(surface,transformed[0],transformed[-1],color,thickness)

    def plot(self,surface,origin,dx,dy,dz,color,thickness=1):
        if len(dx) == len(dy) == len(dz):
            self.drawLine(surface,[[dx[i]+origin[0],-dy[i]+origin[1],-dz[i]+origin[2]] for i in range(len(dx))],color,thickness,0)
        else:
            print("Data arrays need to be the same length")
            raise KeyboardInterrupt

def main(cams):
    img = [0,0]
    img[0] = cams[0].getFrame()
    #img[1] = cams[1].getFrame()

    #Draw a cube
    origin,basis = cams[0].listener.get_hand_data()

    x = -(origin[0]/10.0)
    y = (origin[2]/10.0)
    z = (origin[1]/10.0)-20
    for ii in range(1):
        #vertices = [np.array([i,j,k]) for i in [x-1,x+1] for j in [y-1,y+1] for k in [z-1,z+1]]
        #vertices = cams[ii].newBasis(vertices,basis,origin)

        try:
            #Draw axes
            cams[ii].drawLine(img[ii],cams[ii].newBasis([[x-2,y,z],[x+2,y,z]],basis,origin),(0,0,255),1,1)
            cams[ii].drawLine(img[ii],cams[ii].newBasis([[x,y-2,z],[x,y+2,z]],basis,origin),(0,255,0),1,1)
            cams[ii].drawLine(img[ii],cams[ii].newBasis([[x,y,z-2],[x,y,z+2]],basis,origin),(255,0,0),1,1)

            #Draw some lines or something
            cams[ii].plot(img[ii],[x,y,z],[0,1,2],[0,1,2],[0,1,2],(255,0,255))
        except OverflowError:
            print "oh shit"

    s = int(cams[0].h*(8.0/9.0))
    cams[0].canvas[:,:s] = img[0][:,:s]
    cams[0].canvas[:,s:] = img[0][:,:s]
    return cams[0].canvas

                       
if __name__ == '__main__':
    #main()
    theCamera = Cam(2,(320,240))
    cameraTwo = Cam(0,(320,240))
    theCamera.calibImage()
    cameraTwo.calibImage()
    
