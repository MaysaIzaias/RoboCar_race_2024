#Importação de bibliotecas
import  cv2
import numpy as np

#Abrindo a camera
cap = cv2.VideoCapture(0)


#Criando os sliders de configuração
cv2.namedWindow("Trackbar")

#Declaração da função necessária como parâmetro dos sliders
def nothing(x):
    pass

#Adiciona os sliders para cada parâmetro da máscara de imagem 
cv2.createTrackbar("L-H", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)

cv2.createTrackbar("U-H", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("U-S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("U-V", "Trackbar", 0, 255, nothing)

#Laço de repetição para continuamente capturar a imagem da câmera
while True:
    
    ret, frame = cap.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Definindo os limites de cor
    lh = cv2.getTrackbarPos("L-H","Trackbar")
    ls = cv2.getTrackbarPos("L-S","Trackbar")
    lv = cv2.getTrackbarPos("L-V","Trackbar")
    uh = cv2.getTrackbarPos("U-H","Trackbar")
    us = cv2.getTrackbarPos("U-S","Trackbar")
    uv = cv2.getTrackbarPos("U-V","Trackbar")

    lowerrange = np.array([lh,ls,lv])
    upperrange = np.array([uh,us,uv])

    mask = cv2.inRange(frame, lowerrange,upperrange)

    result = cv2.bitwise_and(frame,frame,mask=mask)

#Abrir as janelas de cada mascara
    cv2.imshow('frame', frame)
    cv2.imshow('HSV', hsv_frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Resultado', result)

#tecla de quebra do codigo (fechar a camera)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()