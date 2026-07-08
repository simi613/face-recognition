import cv2
import numpy as np
from pathlib import Path
import urllib.request
import ssl

class FaceDetector:
    """人脸检测器（支持图片/视频帧+可选性别/年龄/模糊/人脸数量标注）"""
    def __init__(self, use_dnn=True):
        self.use_dnn = use_dnn
        self.has_gender_model = False
        self.has_age_model = False
        self.gender_net = None
        self.age_net = None
        self.gender_list = ["Male", "Female"]
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_mean = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_mean = (78.4263377603, 87.7689143744, 114.895847746)

        if self.use_dnn:
            self._init_dnn()
        else:
            self._init_haar()
        self._init_gender_model()
        self._init_age_model()

    def _init_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Haar人脸分类器加载失败")
        print("[INFO] Haar人脸检测器就绪")

    def _init_dnn(self):
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)
        proto = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        if not proto.exists() or not caffemodel.exists():
            print("[INFO] 下载SSD人脸模型...")
            self._download_face_model(proto, caffemodel)
        if proto.exists() and caffemodel.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
            print("[INFO] DNN SSD人脸检测器就绪")
        else:
            print("[WARN] DNN模型缺失，降级Haar")
            self.use_dnn = False
            self._init_haar()

    def _download_face_model(self, proto_path, model_path):
        ssl._create_default_https_context = ssl._create_unverified_context
        url_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        url_model = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        try:
            urllib.request.urlretrieve(url_proto, str(proto_path))
            urllib.request.urlretrieve(url_model, str(model_path))
        except Exception as e:
            print(f"人脸模型下载失败：{e}")

    def _init_gender_model(self):
        gender_dir = Path(__file__).parent / "gender_models"
        gender_dir.mkdir(exist_ok=True)
        proto = gender_dir / "gender_deploy.prototxt"
        model = gender_dir / "gender_net.caffemodel"
        if not proto.exists() or not model.exists():
            print("[WARN] 性别模型缺失，关闭性别检测")
            self.has_gender_model = False
            return
        try:
            self.gender_net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
            self.has_gender_model = True
            print("[INFO] 性别检测模型加载完成")
        except Exception as e:
            print(f"性别模型加载异常：{e}")
            self.has_gender_model = False

    def _init_age_model(self):
        age_dir = Path(__file__).parent / "age_models"
        age_dir.mkdir(exist_ok=True)
        proto = age_dir / "age_deploy.prototxt"
        model = age_dir / "age_net.caffemodel"
        if not proto.exists() or not model.exists():
            print("[WARN] 年龄模型缺失，关闭年龄检测")
            self.has_age_model = False
            return
        try:
            self.age_net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
            self.has_age_model = True
            print("[INFO] 年龄检测模型加载完成")
        except Exception as e:
            print(f"年龄模型加载异常：{e}")
            self.has_age_model = False

    def _detect_gender(self, frame, faces, enable_gender=True):
        if not self.has_gender_model or not enable_gender:
            return []
        genders = []
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), self.gender_mean, swapRB=False)
            self.gender_net.setInput(blob)
            pred = self.gender_net.forward()
            idx = np.argmax(pred[0])
            genders.append(self.gender_list[idx])
        return genders

    def _detect_age(self, frame, faces, enable_age=True):
        if not self.has_age_model or not enable_age:
            return []
        ages = []
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), self.age_mean, swapRB=False)
            self.age_net.setInput(blob)
            pred = self.age_net.forward()
            idx = np.argmax(pred[0])
            ages.append(self.age_list[idx])
        return ages

    def _blur_faces(self, frame, faces, blur_level=5):
        if blur_level < 1:
            blur_level = 1
        kernel_size = (blur_level * 2 + 1, blur_level * 2 + 1)
        blurred = frame.copy()
        for (x, y, w, h) in faces:
            face_region = blurred[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, kernel_size, 0)
            blurred[y:y+h, x:x+w] = blurred_face
        return blurred

    def detect_faces_image(self, img_path, conf_thresh=0.5, enable_gender=True, enable_age=True):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图片 {img_path}")
        if self.use_dnn:
            cnt, faces, img, confs = self._detect_dnn_frame(img, conf_thresh)
        else:
            cnt, faces, img, confs = self._detect_haar_frame(img)
        genders = self._detect_gender(img, faces, enable_gender)
        ages = self._detect_age(img, faces, enable_age)
        return cnt, faces, img, confs, genders, ages

    def detect_faces_frame(self, frame, conf_thresh=0.5, enable_gender=True, enable_age=True):
        if self.use_dnn:
            cnt, faces, _, confs = self._detect_dnn_frame(frame, conf_thresh)
        else:
            cnt, faces, _, confs = self._detect_haar_frame(frame)
        genders = self._detect_gender(frame, faces, enable_gender)
        ages = self._detect_age(frame, faces, enable_age)
        return cnt, faces, confs, genders, ages

    def _detect_dnn_frame(self, frame, conf_thresh):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104,177,123))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        confs = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf > conf_thresh:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype(int)
                x = max(0, x1)
                y = max(0, y1)
                fw = x2 - x1
                fh = y2 - y1
                faces.append([x,y,fw,fh])
                confs.append(float(conf))
        faces = np.array(faces) if faces else np.array([])
        return len(faces), faces, frame, confs

    def _detect_haar_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20,20))
        return len(faces), faces, frame, []

    def mark_faces(self, frame, faces, confs=None, genders=None, ages=None,
                   enable_blur=False, blur_level=5, enable_count_text=True, enable_gender=True, enable_age=True):
        draw = frame.copy()
        # 人脸模糊
        if enable_blur and len(faces) > 0:
            draw = self._blur_faces(draw, faces, blur_level)

        genders = genders if genders is not None else []
        confs = confs if confs is not None else []
        ages = ages if ages is not None else []

        for idx, (x,y,w,h) in enumerate(faces):
            cv2.rectangle(draw, (x,y), (x+w,y+h), (0,255,0), 2)
            text_parts = []
            # 人脸编号（人脸数量开关控制）
            if enable_count_text:
                text_parts.append(f"Face{idx+1}")
            if idx < len(confs):
                text_parts.append(f"{confs[idx]*100:.1f}%")
            # 性别文字
            if enable_gender and idx < len(genders):
                text_parts.append(genders[idx])
            # 年龄文字
            if enable_age and idx < len(ages):
                text_parts.append(f"Age:{ages[idx]}")
            label = " ".join(text_parts)
            cv2.putText(draw, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return draw

    def process_save_image(self, in_path, out_path, conf_thresh=0.5,
                           enable_gender=True, enable_age=True, enable_blur=False, blur_level=5, enable_count_text=True):
        cnt, faces, img, confs, genders, ages = self.detect_faces_image(in_path, conf_thresh, enable_gender, enable_age)
        marked = self.mark_faces(img, faces, confs, genders, ages,
                                 enable_blur, blur_level, enable_count_text, enable_gender, enable_age)
        cv2.imwrite(str(out_path), marked)
        return cnt, out_path, genders, ages