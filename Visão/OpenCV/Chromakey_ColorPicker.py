import cv2
import numpy as np

colors = []

def on_mouse_click (event, x, y, flags, hsv):
    global colors
    if event == cv2.EVENT_LBUTTONUP:
        #Sets [colors] to the selected pixel HSV values as a LIST var
        colors = hsv[y,x].tolist()

def main():
    #Get the video capturing of webcam/camera
    capture = cv2.VideoCapture(0)

    while True:
        #If "q" is pressed -> exit loop
        if cv2.waitKey(1) == ord('q'):
            break

        #Get the current frame from the camera
        _, frame = capture.read()

        #Converts the frame to HSV format
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #show in the image the selected HSV numbers
        if colors:
            cv2.putText(hsv, str(colors), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        #Show HSV image
        cv2.imshow('frame', hsv)

        #Gets mouse click
        cv2.setMouseCallback('frame', on_mouse_click, hsv)

        #if there is no input on mouse -> start a new loop
        if len(colors) == 0:
            continue

        #Creates Lower and Upper bound numpy arrays
        lower = np.zeros((3,))
        upper = np.zeros((3,))

        #Set interval 
        interval = 20

        for i in range(0,3):
            #If the operation doesn't create a negative number:
            if i > interval:
                lower[i] = colors[i]-interval

            #If the operation doesn't exceed max limit of 255:
            if i < 255-interval: 
                upper[i] = colors[i]+interval

        
        #Create the mask
        mask = cv2.inRange(frame,lowerb=lower,upperb=upper)

        #Apply mask to frame
        result = cv2.bitwise_and(frame,frame,mask=mask) 

        #Show Final image
        cv2.imshow("Resultado",result)

        #Empty list
        colors.clear()

    #Closes the video capturing of webcam/camera
    capture.release()

    #Closes all opened windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()