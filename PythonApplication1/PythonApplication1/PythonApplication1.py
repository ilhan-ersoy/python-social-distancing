import cv2
import imutils
import numpy as np
import argparse
import math

def center(x,w,y,h):
    cx = x + (w / 2)
    cy = y + (h / 2)
    return(cx,cy)

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    person = 1

    toplam_genişlik = 0
    ortalama_genişlik=0
    insan_merkezleri = []

    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        insan_merkezleri.append(center(x,w,y,h))
        toplam_genişlik = toplam_genişlik+w

        cv2.putText(frame, '', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    insansayisi = person
    ortalama_genişlik=toplam_genişlik/insansayisi
    sosyal_mesafe_ihlal = []

    for i, birinci in zip(range(len(insan_merkezleri)), insan_merkezleri):
        for j, ikinci in zip(range(len(insan_merkezleri)), insan_merkezleri[i + 1:]):
            if math.sqrt(((birinci[0] - ikinci[0]) ** 2) + ((birinci[1] - ikinci[1]) ** 2)) < ortalama_genişlik:
                sosyal_mesafe_ihlal.append((i, j + i + 1))



    for (i, j) in sosyal_mesafe_ihlal:
        (x, y, w, h) = bounding_box_cordinates[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        (x, y, w, h) = bounding_box_cordinates[j]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Social Distance Violation', (x, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, 'Warning:', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame,  f'Social Distance Violation:{2*len(sosyal_mesafe_ihlal)}', (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(frame, 'F B U', (35, 380), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, (255, 0, 0), 1)
    cv2.putText(frame, 'Social Distance Program', (35, 410), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
    cv2.imshow('output', frame)


    return frame


def detectByPathVideo(path):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')

    while video.isOpened():
        check, frame = video.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


path = "test.mp4"
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

detectByPathVideo(path)
