import cv2
import numpy as np
import HandTrackingModule as htm

def good(lmList):
    flag = False
    if(len(lmList) != 0):
        for i in range(len(lmList)):
            if(lmList[4][2] > lmList[i][2]):
                flag = True

        return False if flag else True

def bad(lmList):
    flag = False
    if(len(lmList) != 0):
        for i in range(len(lmList)):
            if(lmList[4][2] < lmList[i][2]):
                flag = True

        return False if flag else True


def main():

    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()

    while True:
        success, img = cap.read()
        c = detector.findHands(img)
        lmList = detector.findPosition(img)

        isGood = good(lmList)
        isBad = bad(lmList)

        if(isGood):
            col = (124,252,0)
        elif(isBad):
            col = (128,0,128)
        else:
            col = (0,0,0)

        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, (0, 0), (2000,840), col, cv2.FILLED)

        out = cv2.addWeighted(img, 0.7, blk, 0.3, 0)

        cv2.imshow("Image", out) if (isGood or isBad) else cv2.imshow("Image", img)

        cv2.waitKey(1)


main()

