import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy
import mouse

wCam, hCam = 640, 480  # Kamera çözünürlüğü
wScr, hScr = autopy.screen.size()  # Screen çözünürlüğü

pTime = 0
nosex, nosey = 0, 0
smoothening = 9
frameR = 180  # Mouse frameini daraltmak için verdiğimiz değer

cap = cv2.VideoCapture(0)

prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


def findPosition(img, face_num=0):
    lmList = []
    if results.multi_face_landmarks:
        myFace = results.multi_face_landmarks[face_num]
        for id, lm in enumerate(myFace.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
    return lmList


def findFace(img, draw=True):
    if results.multi_face_landmarks:
        for handLms in results.multi_face_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpFaceMesh.FACEMESH_FACE_OVAL)
    return img


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    img.flags.writeable = False  # Verimlilik arttırır

    img = findFace(img)
    lmList = findPosition(img)

    if len(lmList) != 0:
        # 1 burun ucu
        # 8 kaşların ortası
        # 10 yüzn en tepesi
        # 152 yüzün en aşağısı

        # 13 üst dudak
        # 14 alt dudak
        # 61 dudak sol
        # 291 dudak sağ

        # 159 sol göz üst
        # 52 sol kaş alt
        # 386 sağ göz üst
        # 282 sağ kaş alt

        # Her noktanın kordinatlarını bulduğumuz bölüm
        nosex, nosey = lmList[1][1:]
        up_lipsx, up_lipsy = lmList[13][1:]
        under_lipsx, under_lipsy = lmList[14][1:]
        left_lipsx, left_lipsy = lmList[61][1:]
        right_lipsx, right_lipsy = lmList[291][1:]
        left_eyex, left_eyey = lmList[159][1:]
        left_browx, left_browy = lmList[52][1:]
        right_eyex, right_eyey = lmList[386][1:]
        right_browx, right_browy = lmList[282][1:]

        # üst ve alt dudaklar arası mesafe, sol kaş ve göz arasındaki mesafe, sağ göz ve kaş arasındaki mesafe ve ağzın sağ ve sol uçları arasındaki mesafe hesaplaması
        lips_lenght = math.hypot(up_lipsx - under_lipsx, up_lipsy - under_lipsy)  # 2 - 50
        left_eye_brow_lenght = math.hypot(left_eyex - left_browx, left_eyey - left_browy)  # 25 - 42
        right_eye_brow_lenght = math.hypot(right_eyex - right_browx, right_eyey - right_browy)  # 25 - 42
        left_right_lips_length = math.hypot(left_lipsx - right_lipsx, left_lipsy - right_lipsy)
        cv2.circle(img, (nosex, nosey), 5, (255, 0, 255), cv2.FILLED)

        # Mouse hareketleri ve smoothening yapıldığı kısım
        x3 = np.interp(nosex, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(nosey, (frameR, hCam - frameR), (0, hScr))

        currLocX = prevLocX + (x3 - prevLocX) / smoothening
        currLocY = prevLocY + (y3 - prevLocY) / smoothening

        mouse.move(wScr - currLocX, currLocY)
        prevLocX, prevLocY = currLocX, currLocY

        # Kaşları kaldırınca tıkla kısa bir süre kalkık tutunca birden çok defa tıkla
        if left_eye_brow_lenght > 26 and right_eye_brow_lenght > 26:
            mouse.click()
            time.sleep(0.26)

        # gülümseyince sağ tıkla
        if left_right_lips_length > 75:
            mouse.right_click()
            time.sleep(0.18)

        # Ağzı açınca basılı tutmaya başla bırakmak için bir kere sol tıkla
        if lips_lenght > 32:
            mouse.press()
            time.sleep(0.18)

    # FPS gösterildiği kısım
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    img = cv2.flip(img, 1)  # Kameramıza mirror efekti uyguladık
    cv2.putText(img, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
