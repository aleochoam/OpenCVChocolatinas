from imutils import resize
from time import time
import numpy as np
import cv2
import argparse

contador = dict()

contador["nr_jet_azul"] = 0
contador["nr_flow_negra"] = 0
contador["nr_flow_blanca"] = 0
contador["nr_jumbo_naranja"] = 0
contador["nr_jumbo_roja"] = 0
contador["nr_chocorramo"] = 0
contador["nr_frunas_verde"] = 0
contador["nr_frunas_naranja"] = 0
contador["nr_frunas_roja"] = 0
contador["nr_frunas_amarilla"] = 0
contador["tt"] = 0

kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

lim_colores = [
    ([88, 31, 0], [130, 255, 255], "nr_jet_azul"), #Azul
    ([0, 0, 0], [0, 75, 65], "nr_flow_negra"), #Negro
    ([0, 0, 100], [240, 3, 100], "nr_flow_blanca"), #Blanco
    ([0, 0, 0], [37, 255, 255], "nr_jumbo_roja"), #Rojo
    #jumbo naranja
    #jumbo roja
    #chocorramo
    ([120, 75, 80], [120, 100, 100], "nr_frunas_verde"),  # Verde
    #frunas naranja
    #frunas roja
    ([20, 160, 160], [50, 255, 255], "nr_frunas_amarilla"), #Amarillo

    # ([33, 100, 100], [39, 100, 100], "milo"), #Naranja
    ]


def contar_blancos(roi):
    return cv2.countNonZero(roi)


def generar_mascara(lower, upper, roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # res = cv2.bitwise_and(roi, roi, mask=mask)
    return mask


def find_centroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to file")

    return vars(ap.parse_args())


def main(args):

    if args.get("video", None) is None:
        capture = cv2.VideoCapture(1)
    else:
        capture = cv2.VideoCapture(args.get("video"))

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    found = False
    cont_chocolatinas = 0

    tiempo_desactivado = time()
    while True:
        grabbed, frame = capture.read()

        if not grabbed:
            # End of Video
            break

        frame = resize(frame, width=500)

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

            if not found and cx > frame.shape[1]/2:
                roi = frame[y:y+h, x:x+w]

                found = True
                if roi is not None:
                    max_blancos = 0
                    nombre_mascara = ""

                    for lower, upper, nombre in lim_colores:
                        lower = np.array(lower, dtype="uint8")
                        upper = np.array(upper, dtype="uint8")
                        mask = generar_mascara(lower, upper, roi)

                        blancos_mascara = contar_blancos(mask)
                        if max_blancos < blancos_mascara:
                            max_blancos = blancos_mascara
                            nombre_mascara = nombre

                    contador[nombre_mascara] = contador[nombre_mascara] + 1
                    print(nombre_mascara)
                    cv2.imshow("roi", roi)

        # Si no hay contornos
        else:

            # Se verifica que no hayan transcurrido 3 segundos
            if abs(tiempo_desactivado - time()) > 3.0:
                break
            found = False
            roi = None

        dist_top = 10
        for key, value in contador.items():
            texto = str(key) + ": " + str(value)
            cv2.putText(frame, texto, (10, dist_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            dist_top = dist_top + 20


        cv2.imshow("Original", frame)
        cv2.imshow("azul", generar_mascara(np.array(lim_colores[0][0]), np.array(lim_colores[0][1]), frame))
        cv2.imshow("rojo", generar_mascara(np.array(lim_colores[3][0]), np.array(lim_colores[3][1]), frame))

        # Escape para terminar
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tiempoInicial = time()
    main(parse_arguments())
    contador["tt"] = time() - tiempoInicial
    print(contador)
