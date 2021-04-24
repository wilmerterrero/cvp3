import cv2
import os
import imutils
import numpy as np

class Recognize:
    dataPath = 'Rostro reconocido'
    imagePaths = None
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    faceClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades+'data/haarcascades/haarcascade_frontalface_default.xml')

    def SetDataPath():
        if not os.path.exists(Recognize.dataPath):
            os.makedirs(Recognize.dataPath)

    def SetData():
        Recognize.SetDataPath()
        Recognize.imagePaths = os.listdir(Recognize.dataPath)
        Recognize.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        if os.path.exists('data/haarcascades/EigenFace.xml'):
            Recognize.face_recognizer.read('data/haarcascades/EigenFace.xml')
        Recognize.faceClassif = cv2.CascadeClassifier(
            cv2.data.haarcascades+'data/haarcascades/haarcascade_frontalface_default.xml')

    def SaveFace(cap):
        Recognize.SetDataPath()

        faceClassif = cv2.CascadeClassifier(
            cv2.data.haarcascades+'data/haarcascades/haarcascade_frontalface_default.xml')
        count = 0

        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = imutils.resize(frame, width=900)
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()
            faces = faceClassif.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), 1)
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150),
                                    interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(Recognize.dataPath +
                            '/rotro_{}.jpg'.format(count), rostro)
                count = count + 1
            cv2.imshow('Guardando rostro', frame)
            k = cv2.waitKey(1)
            if k == 27 or count >= 100:
                break
        cv2.destroyWindow('Guardando rostro')

    def TrainModel():
        peopleList = os.listdir(Recognize.dataPath)
        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = Recognize.dataPath + '/' + nameDir
            labels.append(label)
            facesData.append(cv2.imread(personPath, 0))
            label = label + 1

        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write('data/haarcascades/EigenFace.xml')

    def RecognizeFace(cap):
        Recognize.SetData()
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = imutils.resize(frame, width=900)
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()

            faces = Recognize.faceClassif.detectMultiScale(gray, 1.3, 5)
            cv2.putText(frame, 'Presione G para reconocer gestos',
                        (0, 610), 2, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Presione S para guardar rostro',
                        (0, 640), 2, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Presione ESC para salir', (0, 670),
                        2, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            for (x, y, w, h) in faces:
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150),
                                    interpolation=cv2.INTER_CUBIC)
                result = (0, 5700)
                if os.path.exists('data/haarcascades/EigenFace.xml'):
                    result = Recognize.face_recognizer.predict(rostro)
                cv2.putText(frame, '{}'.format(result), (x, y-5),
                            1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                # EigenFaces
                if result[1] < 5700:
                    cv2.putText(frame, 'Rostro conocido', (x, y-25),
                                2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Desconocido', (x, y-20), 2,
                                0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            cv2.imshow('Reconomiento facial', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('s'):
                Recognize.SaveFace(cap)
                Recognize.TrainModel()
                Recognize.SetData()
            elif k == ord('g'):
                Recognize.Signs(cap)

    def Signs(cap):
        bg = None

        # Ingresamos el algoritmo
        faceClassif = cv2.CascadeClassifier(
            cv2.data.haarcascades+'data/haarcascades/haarcascade_frontalface_default.xml')
        color_contorno = (0, 255, 0)
        color_ymin = (0, 130, 255)
        color_fingers = (0, 255, 255)

        while True:
            ret, frame = cap.read()
            if ret == False:
                break

            # Redimensionar la imagen para que tenga un ancho de 640
            frame = imutils.resize(frame, width=900)
            frame = cv2.flip(frame, 1)
            frameAux = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            cv2.putText(frame, 'Presione Q para volver a tomar el fondo',
                        (0, 645), 2, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Presione ESC para salir', (0, 670),
                        2, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Detectando dedos
            if bg is not None:

                # Determinar la región de interés
                ROI = frame[200:600, 460:800]
                cv2.rectangle(frame, (460-2, 200-2),
                              (800+2, 600+2), color_fingers, 1)
                grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

                # Región de interés del fondo de la imagen
                bgROI = bg[200:600, 460:800]

                # Determinar la imagen binaria (background vs foreground)
                dif = cv2.absdiff(grayROI, bgROI)
                _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
                th = cv2.medianBlur(th, 7)

                # Encontrando los contornos de la imagen binaria
                cnts, _ = cv2.findContours(
                    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

                for cnt in cnts:

                    # Encontrar el centro del contorno
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        M["m00"] = 1
                    x = int(M["m10"]/M["m00"])
                    y = int(M["m01"]/M["m00"])
                    cv2.circle(ROI, tuple([x, y]), 5, (0, 255, 0), -1)

                    # Punto más alto del contorno
                    ymin = cnt.min(axis=1)
                    cv2.circle(ROI, tuple(ymin[0]), 5, color_ymin, -1)

                    # Contorno encontrado a través de cv2.convexHull
                    hull1 = cv2.convexHull(cnt)
                    cv2.drawContours(ROI, [hull1], 0, color_contorno, 2)

                    # Defectos convexos
                    hull2 = cv2.convexHull(cnt, returnPoints=False)
                    defects = cv2.convexityDefects(cnt, hull2)

                    # Seguimos con la condición si es que existen defectos convexos
                    if defects is not None:

                        inicio = []  # Contenedor en donde se almacenarán los puntos iniciales de los defectos convexos
                        fin = []  # Contenedor en donde se almacenarán los puntos finales de los defectos convexos
                        fingers = 0  # Contador para el número de dedos levantados

                        for i in range(defects.shape[0]):

                            s, e, f, d = defects[i, 0]
                            start = cnt[s][0]
                            end = cnt[e][0]
                            far = cnt[f][0]

                            # Encontrar el triángulo asociado a cada defecto convexo para determinar ángulo
                            a = np.linalg.norm(far-end)
                            b = np.linalg.norm(far-start)
                            c = np.linalg.norm(start-end)

                            angulo = np.arccos(
                                (np.power(a, 2)+np.power(b, 2)-np.power(c, 2))/(2*a*b))
                            angulo = np.degrees(angulo)
                            angulo = int(angulo)

                            # Se descartarán los defectos convexos encontrados de acuerdo a la distnacia
                            # entre los puntos inicial, final y más alelago, por el ángulo y d
                            if np.linalg.norm(start-end) > 20 and angulo < 90 and d > 12000:

                                # Almacenamos todos los puntos iniciales y finales que han sido
                                # obtenidos
                                inicio.append(start)
                                fin.append(end)

                        # Si no se han almacenado puntos de inicio (o fin), puede tratarse de
                        # 0 dedos levantados o 1 dedo levantado
                        if len(inicio) == 0:
                            minY = np.linalg.norm(ymin[0]-[x, y])
                            if minY >= 110:
                                fingers = fingers + 1

                        for i in range(len(inicio)):
                            fingers = fingers + 1
                            if i == len(inicio)-1:
                                fingers = fingers + 1

                        if fingers >= 0:

                            if fingers == 0:
                                cv2.putText(frame, '{} VIOLENCIA'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                            elif fingers == 1:
                                cv2.putText(frame, '{} ACOSO'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                            elif fingers == 2:
                                cv2.putText(frame, '{} VIOLENCIA DOMESTICA'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                            elif fingers == 3:
                                cv2.putText(frame, '{} INCENDIO'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                            elif fingers == 4:
                                cv2.putText(frame, '{} PRIMEROS AUXILIOS'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                            elif fingers == 5:
                                cv2.putText(frame, '{} INUNDACION'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, '{} dedos levantados'.format(
                                    fingers), (0, 45), 1, 2, (color_fingers), 2, cv2.LINE_AA)
                    else:
                        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)

            k = cv2.waitKey(20)
            if k == ord('q'):
                bg = None
            if k == 27:
                break
            cv2.imshow('Signs', frame)
        cv2.destroyWindow('Signs')
