import cv2
from imutils import resize
import argparse

cont

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
    cont_chocolatinas = 0

    while True:
        grabbed, frame = capture.read()

        if not grabbed:
            # End of Video
            break

        frame = resize(frame, width=500)
        thresh = fgbg.apply(frame)

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        if len(contours) > 0 and cv2.contourArea(contours[0]) > 200:
            cnt = contours[0]
            (x, y, w, h) = cv2.boundingRect(cnt)
            cx, cy = find_centroid(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0))

            if not found:
                cont_chocolatinas = cont_chocolatinas+1
                found = True
        else:
            found = False

        cv2.putText(frame, "Chocolatinas encontradas: {}".format(cont_chocolatinas), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Original", frame)
        # cv2.imshow("ROI", )

        # Escape para terminar
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parse_arguments())
