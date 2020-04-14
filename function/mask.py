import paddlehub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont #与cv2 进行转换PIl可以显示汉字cv2不行
import cv2
module = hub.Module(name="pyramidbox_lite_server_mask", version='1.2.0')
camera = cv2.VideoCapture(0)

def mask_detection(frame):
    # 改变摄像头图像的大小，图像小，所做的计算就少
    _,frame=camera.read()
    frame_copy = frame.copy()
    input_dict = {"data": [frame_copy]}
    # 口罩检测
    results = module.face_detection(data=input_dict)
    for result in results:
        labelmask = result['data']['label']
        confidence_origin = result['data']['confidence']
        confidence = round(confidence_origin, 2)
        confidence_desc = str(confidence)
        top, right, bottom, left = int(result['data']['top']), int(
            result['data']['right']), int(result['data']['bottom']), int(
            result['data']['left'])
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 6), 5)
        print('mask detection is running')
        if labelmask == 'NO MASK':
            label_cn = '无口罩'
        if labelmask == 'MASK':
            label_cn = '有口罩'
        # 由于opencv无法显示汉字之前使用的方法当照片很小时会报错，此次采用了另一种方法使用PIL进行转换
        cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype("webvideo/function/msyh.ttf", 27, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text((left + 10, bottom), '状态：' + label_cn, (255, 255, 255),
                  font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        draw.text((left + 10, bottom + 30), '体温：36.5℃', (255, 255, 255), font=font)
        # PIL图片转cv2图片
        frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

        # yield cv2.imencode('.jpg', frame)[1].tobytes()
        return frame
    # print(frame)
