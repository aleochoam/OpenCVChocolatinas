from imutils import resize
from time import time, sleep
from tkinter import Tk, Button, Label
import numpy as np
import cv2
import argparse
import serial

contador = dict()

contador["nr_jet_azul"] = 0
contador["nr_flow_negra"] = 0
contador["nr_flow_blanca"] = 0
contador["nr_jumbo_naranja"] = 0
contador["nr_jumbo_roja"] = 0
contador["tt"] = 0

kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

lim_colores = [
    ([88, 31, 0], [130, 255, 255], "nr_jet_azul"),
    ([0, 90, 85],   [26, 255, 255], "nr_flow_negra"),
    ([0, 0, 210], [10, 20, 255],   "nr_flow_blanca"),
    ([11, 0, 214], [34, 255, 255], "nr_jumbo_naranja"),
    ([0, 106, 0],  [17, 255, 255],  "nr_jumbo_roja"),
    ]

lim_colores_dia = [
    ([88, 31, 0], [130, 255, 255], "nr_jet_azul"),
    ([8, 16, 83],   [25, 192, 161], "nr_flow_negra"),
    ([0, 0, 146], [170, 20, 255],   "nr_flow_blanca"),
    ([10, 76, 179], [17, 255, 255], "nr_jumbo_naranja"),
    ([0, 120, 157],  [9, 255, 255],  "nr_jumbo_roja"),
    ]


def actualizar_contador():

    texto = "nr_jet_azul" + ": " + str(contador["nr_jet_azul"]) + "\n"
    texto = texto + "nr_flow_negra" + ": " + str(contador["nr_flow_negra"]) + "\n"
    texto = texto + "nr_flow_blanca" + ": " + str(contador["nr_flow_blanca"]) + "\n"
    texto = texto + "nr_jumbo_naranja" + ": " + str(contador["nr_jumbo_naranja"]) + "\n"
    texto = texto + "nr_jumbo_roja" + ": " + str(contador["nr_jumbo_roja"]) + "\n"

    lbl["text"] = texto

    # for key, value in contador.items():
    #     if key == "tt":
    #         continue
    #     item = key + ": " + str(value) + "\n"
    #     texto = texto + item
    lbl["text"] = texto


def contar_blancos(roi):
    return cv2.countNonZero(roi)


def generar_mascara(lower, upper, roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # res = cv2.bitwise_and(roi, roi, mask=mask)
    return mask


def find_centroid(contour):
    m = cv2.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])

    return cx, cy


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to file")

    return vars(ap.parse_args())


def clasificar():
    args = parse_arguments()
    if args.get("video", None) is None:
        capture = cv2.VideoCapture(1)
    else:
        capture = cv2.VideoCapture(args.get("video"))

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    found = False

    tiempo_desactivado = time()
    while True:
        grabbed, frame = capture.read()

        if not grabbed:
            # End of Video
            break

        frame = resize(frame, width=500)
        frame = frame[50:3500, 75:400]

        thresh = fgbg.apply(frame)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOP, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelCL, iterations=2)

        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        # Si se encuentra un contorno en la imagen
        if len(contours) > 0 and cv2.contourArea(contours[0]) > 3000:
            tiempo_desactivado = time()
            cnt = contours[0]
            (x, y, w, h) = cv2.boundingRect(cnt)
            cx, cy = find_centroid(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0))

            if not found and cy > frame.shape[0]/2:
                roi = frame[y:y+h, x:x+w]

                found = True
                if roi is not None:
                    cv2.imshow("roi", roi)
                    max_blancos = 0
                    nombre_mascara = ""

                    for lower, upper, nombre in lim_colores_dia:
                        lower = np.array(lower, dtype="uint8")
                        upper = np.array(upper, dtype="uint8")
                        mask = generar_mascara(lower, upper, roi)

                        blancos_mascara = contar_blancos(mask)
                        #print(nombre, blancos_mascara)
                        #cv2.imshow("mascara", mask)

                        #cv2.waitKey(0)
                        if max_blancos < blancos_mascara:
                            max_blancos = blancos_mascara
                            nombre_mascara = nombre

                    contador[nombre_mascara] = contador[nombre_mascara] + 1
                    actualizar_contador()
                    root.update_idletasks()
                    print(nombre_mascara)
                    # print("--------------")

        # Si no hay contornos
        else:

            # Se verifica que no hayan transcurrido 3 segundos
            if abs(time() - tiempo_desactivado) > 3.0:
                break
            found = False
            # roi = None

        cv2.imshow("Original", frame)
        # cv2.imshow("thresh", thresh)

        # Escape para terminar
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    finalizar_arduino()
    capture.release()
    cv2.destroyAllWindows()
    # root.destroy()


def iniciar_arduino():
    global arduino
    print("Inicializando arduino")

    com = "/dev/ttyACM0"
    arduino = serial.Serial(com, 9600)
    sleep(2)
    arduino.write(b's')


def finalizar_arduino():
    arduino.write(b'f')
    arduino.close()


def main():
    btn.destroy()
    iniciar_arduino()

    tiempo_inicial = time()
    clasificar()

    contador["tt"] = time() - tiempo_inicial
    actualizar_contador()
    print(contador)


if __name__ == '__main__':
    arduino = None
    root = Tk()
    btn = Button(root, text="Start", command=main)
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    lbl = Label(root, text="")
    lbl.pack(side="left")
    actualizar_contador()
    root.mainloop()
