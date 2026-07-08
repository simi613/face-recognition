"""
人脸识别Web服务 - 文件名匹配人名版（修复文件类型校验报错，移除filetype校验兼容Windows）
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import cv2
import os

app = FastAPI(title="AI人脸识别-文件名匹配人名", version="3.0")
BASE_DIR = Path(__file__).parent
STATIC_IMG = BASE_DIR / "static" / "images"
STATIC_IMG.mkdir(parents=True, exist_ok=True)
NAME_MAPPING_DIR = BASE_DIR / "name_mapping"
NAME_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# 基础图片格式，移除heic避免解码问题
ALLOW_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
MAX_SIZE = 10 * 1024 * 1024

# 人脸检测器
from face_detector import FaceDetector
detector = FaceDetector(use_dnn=False)
camera = None
realtime_switch = False
realtime_conf = 0.5

# ---------------------- 文件名匹配人名逻辑 ----------------------
def load_name_file_library():
    name_set = set()
    for img_path in NAME_MAPPING_DIR.glob("*"):
        if not img_path.is_file():
            continue
        suffix = img_path.suffix.lower()
        if suffix not in ALLOW_EXT:
            continue
        person_name = img_path.stem.strip()
        name_set.add(person_name)
    return name_set

def get_match_name(upload_file_name: str):
    name_lib = load_name_file_library()
    clean_name = upload_file_name.strip().rstrip(".")
    upload_stem = Path(clean_name).stem.strip()
    if upload_stem in name_lib:
        return upload_stem
    return "未知"

# ---------------------- 工具函数：清洗文件名后缀 ----------------------
def clean_filename(filename: str) -> str:
    return filename.strip().rstrip(".")

# ---------------------- 路由接口 ----------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    tpl = TEMPLATE_DIR / "index.html"
    if not tpl.exists():
        raise HTTPException(404, "templates/index.html 不存在")
    with open(tpl, "r", encoding="utf-8") as f:
        return f.read()

# ====================== 修改后的完整 upload 接口 ======================
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    enable_gender: bool = Form(True),
    enable_age: bool = Form(True),
    enable_blur: bool = Form(False),
    blur_level: int = Form(5),
    enable_count_text: bool = Form(True)
):
    # 读取文件字节
    data = await file.read()
    # 文件大小校验
    if len(data) > MAX_SIZE:
        raise HTTPException(400, "文件超过10MB上限")

    # 清洗文件名，提取后缀（仅后缀校验，移除filetype二进制校验）
    clean_name = clean_filename(file.filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOW_EXT:
        raise HTTPException(400, f"当前不支持该文件类型，请尝试其他文件，支持格式：{','.join(ALLOW_EXT)}")

    # 生成临时文件
    uid = str(uuid.uuid4())
    tmp_in = STATIC_IMG / f"tmp_{uid}{ext}"
    with open(tmp_in, "wb") as f:
        f.write(data)

    try:
        out_name = f"res_{uid}{ext}"
        out_path = STATIC_IMG / out_name
        # 人脸处理
        count, _, genders, ages = detector.process_save_image(
            tmp_in, out_path, confidence,
            enable_gender, enable_age, enable_blur, blur_level, enable_count_text
        )
        person_name = get_match_name(file.filename)
        male = genders.count("Male")
        female = genders.count("Female")
        msg = f"检测到{count}张人脸"
        if enable_gender and count > 0:
            msg += f"，男性{male}，女性{female}"
        if enable_blur and count > 0:
            msg += "，已人脸模糊处理"
        if not enable_count_text:
            msg += "，隐藏人脸编号标注"
        if not enable_gender:
            msg += "，关闭性别检测"
        if not enable_age:
            msg += "，关闭年龄检测"
        msg += f"，匹配人名：{person_name}"

        return {
            "success": True,
            "face_count": count,
            "genders": genders if enable_gender else [],
            "ages": ages if enable_age else [],
            "result_filename": out_name,
            "message": msg,
            "model": "DNN SSD" if detector.use_dnn else "Haar",
            "person_name": person_name
        }
    except ValueError as e:
        raise HTTPException(400, f"图片损坏无法打开：{str(e)}")
    finally:
        # 删除临时文件
        if tmp_in.exists():
            tmp_in.unlink()
# =====================================================================

@app.get("/result/{fname}")
async def get_img(fname: str):
    p = STATIC_IMG / fname
    if not p.exists():
        raise HTTPException(404, "图片不存在")
    return FileResponse(p)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gender_detect": detector.has_gender_model,
        "age_detect": detector.has_age_model
    }

# 摄像头原始流
def gen_raw_stream():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise HTTPException(500, "摄像头打开失败")
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        _, buf = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

# 实时检测流
def gen_realtime_stream():
    global camera, realtime_switch, realtime_conf
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise HTTPException(500, "摄像头打开失败")
    while realtime_switch:
        ret, frame = camera.read()
        if not ret:
            break
        face_cnt, faces, confs, genders, ages = detector.detect_faces_frame(frame, realtime_conf)
        draw_frame = detector.mark_faces(frame, faces, confs, genders, ages)
        m = genders.count("Male")
        f = genders.count("Female")
        cv2.putText(draw_frame, f"Faces:{face_cnt} Male:{m} Female:{f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode(".jpg", draw_frame)
        yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

@app.get("/camera/stream")
async def raw_cam():
    return StreamingResponse(gen_raw_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/realtime_stream")
async def realtime_cam(confidence: float = 0.5):
    global realtime_switch, realtime_conf
    realtime_switch = True
    realtime_conf = confidence
    return StreamingResponse(gen_realtime_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/stop_realtime")
async def stop_rt():
    global realtime_switch
    realtime_switch = False
    return {"msg": "实时检测已停止"}

@app.get("/camera/capture")
async def capture(
    confidence: float = 0.5,
    enable_gender: bool = True,
    enable_age: bool = True,
    enable_blur: bool = False,
    blur_level: int = 5,
    enable_count_text: bool = True
):
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise HTTPException(500, "摄像头打开失败")
    ret, frame = camera.read()
    if not ret:
        raise HTTPException(500, "读取画面失败")
    uid = str(uuid.uuid4())
    tmp = STATIC_IMG / f"cap_tmp_{uid}.jpg"
    cv2.imwrite(str(tmp), frame)
    try:
        out_name = f"cap_res_{uid}.jpg"
        out_p = STATIC_IMG / out_name
        cnt, _, genders, ages = detector.process_save_image(
            tmp, out_p, confidence,
            enable_gender, enable_age, enable_blur, blur_level, enable_count_text
        )
        person_name = "未知"
        m = genders.count("Male")
        f = genders.count("Female")
        msg = f"检测到{cnt}张人脸"
        if enable_gender and cnt > 0:
            msg += f"，男性{m}，女性{f}"
        if enable_blur and cnt > 0:
            msg += "，已人脸模糊处理"
        if not enable_count_text:
            msg += "，隐藏人脸编号标注"
        if not enable_gender:
            msg += "，关闭性别检测"
        if not enable_age:
            msg += "，关闭年龄检测"
        msg += f"，匹配人名：{person_name}"
        return {
            "success": True,
            "face_count": cnt,
            "genders": genders if enable_gender else [],
            "ages": ages if enable_age else [],
            "result_filename": out_name,
            "message": msg,
            "person_name": person_name
        }
    except ValueError as e:
        raise HTTPException(400, f"画面处理失败：{str(e)}")
    finally:
        if tmp.exists():
            tmp.unlink()

@app.get("/camera/close")
async def close_cam():
    global camera, realtime_switch
    realtime_switch = False
    if camera:
        camera.release()
        camera = None
    return {"msg": "摄像头已释放"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)