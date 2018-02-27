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
contador["tt"] = 0

kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

lim_colores = [
    ([88, 31, 0], [130, 255, 255], "nr_jet_azul"),
    ([0, 90, 85],   [26, 255, 255],"nr_flow_negra"),
    ([0, 0, 210], [10, 20, 255],   "nr_flow_blanca"),
    ([11, 0, 214], [34, 255, 255], "nr_jumbo_naranja"), #Naranja
    ([0, 106, 0],  [17, 255, 255],  "nr_jumbo_roja"),
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
        capture = cv2.VideoCapture(0)
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
            print(cv2.contourArea(contours[0]))
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
                    cv2.imshow("roi", roi)
                    max_blancos = 0
                    nombre_mascara = ""

                    for lower, upper, nombre in lim_colores:
                        lower = np.array(lower, dtype="uint8")
                        upper = np.array(upper, dtype="uint8")
                        mask = generar_mascara(lower, upper, roi)

                        blancos_mascara = contar_blancos(mask)
                        print(nombre, blancos_mascara)
                        cv2.imshow("mascara", mask)

                        cv2.waitKey(0)
                        if max_blancos < blancos_mascara:
                            max_blancos = blancos_mascara
                            nombre_mascara = nombre

                    contador[nombre_mascara] = contador[nombre_mascara] + 1
                    print(nombre_mascara)
                    print("--------------")

        # Si no hay contornos
        else:

            # Se verifica que no hayan transcurrido 3 segundos
            if abs(time() - tiempo_desactivado) > 10.0:
                break
            found = False
            roi = None

        dist_top = 10
        for key, value in contador.items():
            if key == "tt":
                continue

            texto = str(key) + ": " + str(value)
            cv2.putText(frame, texto, (10, dist_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            dist_top = dist_top + 20

        cv2.imshow("Original", frame)
        cv2.imshow("thresh", thresh)

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
