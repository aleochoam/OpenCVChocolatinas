import cv2
from imutils import resize
import numpy as np
import argparse


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to file")

    args = vars(ap.parse_args())

    if args.get("video", None) is None:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(args.get("video"))

    first_frame = None

    while True:
        grabbed, frame = capture.read()

        if not grabbed:
            # End of Video
            break

        frame = resize(frame, width=500)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if first_frame is None:
            first_frame = frame

        frameDelta = cv2.absdiff(frame, first_frame)
        _, thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        im2, conts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thresh, conts, -1, (0, 255, 255))

        cv2.imshow("Original", frame)
        cv2.imshow("Resta", thresh)


        # Escape para terminar
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()