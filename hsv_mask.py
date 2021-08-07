import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
import imutils
import mouse
import pyautogui

videoCaptureObject = cv2.VideoCapture(0)

pyautogui.FAILSAFE = False
def getCapturedImage():
    
    while(True):
        ret, frame = videoCaptureObject.read()
        cv2.imshow('Capturing Video', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('space')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            return frame 
    





def nothing(x):
    pass
    
def trackbar_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_LAB2LRGB)


    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
        # mask2 = cv2.inRange(image2, lower, upper)
        # result2 = cv2.bitwise_and(image2, image2, mask=mask2)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q') or keyboard.is_pressed('space'):
            return [hMin, sMin, vMin, hMax, sMax, vMax, result]

    cv2.destroyAllWindows()
    
def getMaskedFrame(hsv_list, frame_input):
    frame_input = cv2.cvtColor(frame_input, cv2.COLOR_LAB2LRGB)
    # Set minimum and maximum HSV values to display
    lower = np.array([hsv_list[0], hsv_list[1], hsv_list[2]])
    upper = np.array([hsv_list[3], hsv_list[4], hsv_list[5]])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(frame_input, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_input, lower, upper)
    result = cv2.bitwise_and(frame_input, frame_input, mask=mask)
    
    return result

def pointer_algo(x,y):
    
    if x <= 0:
        x=0
    if y <= 0:
        y=0
    
    return (x,y)

def apply_contours(image, intial_img):
    ###########################################
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            cX = int(M["m10"] / 0.01)
            cY = int(M["m01"] / 0.01)
        # draw the contour and center of the shape on the image
        cv2.drawContours(intial_img, [c], -1, (0, 255, 0), 2)
        cv2.circle(intial_img, (cX, cY), 7, (255, 255, 255), -1)
        algo_result = pointer_algo(x=cX,y=cY)
        print('Centroid is------->',algo_result[0]," ", algo_result[1])
        pyautogui.moveTo(cX,cY,duration=0)
        cv2.putText(intial_img, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # cv2.imshow("Image", intial_img)
    # return the image
    return intial_img


def main():
    print('Captured image')
    img = getCapturedImage()
    cv2.imshow('Captured Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vidObj = cv2.VideoCapture(0)

    hsv_list = trackbar_hsv(img)
    
    while(True):
        ret, frame = vidObj.read()
        cv2.flip(frame,1)
        cv2.imshow('Capturing Video', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('space')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
        maskedFrame = getMaskedFrame(hsv_list, frame)
        contoured_img = apply_contours(maskedFrame,img)
        cv2.imshow('Capturing Video', contoured_img)
        ##Mouse functions
        
        
    


main()





    


