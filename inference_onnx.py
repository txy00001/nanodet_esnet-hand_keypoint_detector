import time

import cv2
import numpy as np
import onnxruntime
import cv2 as cv
from torchvision import transforms as T
import os

np.set_printoptions(suppress=True)

def resizeAndPad(img, size, padColor=127):
    """
    credit: https://stackoverflow.com/a/49406095/12169382
    """
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv.INTER_NEAREST

    else:  # stretching image
        interp = cv.INTER_LINEAR

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(
            int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(
            int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    # color image but only one color provided
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)

    return scaled_img


def onnx_image_example():
    # -----------------common part 1---------------------
    model_path = r"E:\porject\onnx_infer\out.onnx"
    session_option = onnxruntime.SessionOptions()
    # session_option.optimized_model_filepath = f"{model_file_name}_cudaopt.onnx"
    session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_option.log_severity_level = 3
    # onnxruntime.capi._pybind_state.set_openvino_device('CPU_FP32')
    session = onnxruntime.InferenceSession(
        model_path,
        session_option,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print("==>> input_name: ", input_name)  # input
    output_names = [o.name for o in session.get_outputs()]
    print("==>> output_names: ", output_names)
    input_shape = session.get_inputs()[0].shape
    print("==>> input_shape: ", input_shape)



    img = cv.imread(r"data\img.png")


    resized_imgs = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
    # print("==>> resized_img.shape: ", resized_imgs.shape)
    resized_img = np.transpose(resized_imgs, (2, 0, 1))
    # print("==>> trans_img.shape: ", resized_img.shape)
    resized_img = np.expand_dims(resized_img, axis=0)
    # print("==>> expand_img.shape: ", resized_img.shape)
    dummy_input = np.array(resized_img).astype(np.float32) / 255.0
    dummy_input = (dummy_input - 0.5433) / 0.2005



    output = session.run(
        output_names,
        {input_name: dummy_input}
    )
    s1 = time.time()
    for i in range(1):
      pred_onnx = session.run(output_names,
        {input_name: dummy_input})

    # resized_imgs= resized_img.astype(np.uint8)
    cv.putText(resized_imgs.astype(np.float32), "DetectionNet", (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 显示模型名
    t1 = time.time()
    cv.imshow("Save", resized_imgs)
    cv.waitKey(0)
    print("用onnx完成1次推理消耗的时间:%s" % (t1-s1))

    #print(pred_onnx[0].tolist())






if __name__ == "__main__":
    onnx_image_example()


