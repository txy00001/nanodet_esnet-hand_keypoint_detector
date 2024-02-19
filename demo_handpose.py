import argparse
import os
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import torch
import numpy as np
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir
import onnxruntime

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]

# joint_pairs = [[0, 1],[1, 2], [2, 3], [3, 4],
#                    [0, 5],[5, 6], [6, 7], [7, 8],
#                    [9, 10],[10, 11], [11, 12],
#                    [13, 14],[14, 15], [15, 16],
#                    [0,17],[17,18],[18,19],[19,20],
#                    [5,9],[9,13],[13,17]]
# joint_colors = [(255,0,0),(255,0,0), (255,0,0), (255,0,0),
#                 (125,125,125),(125,125,125), (125,125,125), (125,125,125),
#                 (255,10,200),(255,10,200), (255,10,200),
#                 (100,110,210),(100,110,210), (100,110,210),
#                 (200,10,100),(200,10,100), (200,10,100), (200,10,100),
#                 (169,40,248),(126,0,180), (190,140,255)]


joint_pairs = [[0, 1],[1, 2],
                   [0, 5],


                   [0,17],
                   [5,9],[9,13],[13,17]]
joint_colors = [(255,0,0),(255,0,0),
                (125,125,125),


                (200,10,100),
                (169,40,248),(126,0,180), (190,140,255)]

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans_dst_src = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans_src_dst = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans_src_dst,trans_dst_src

def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class HandPose():
    def __init__(self,model_path) -> None:
        self.net_input_w = 224
        self.net_input_h = 224
        self.model = onnxruntime.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name_xy = self.model.get_outputs()[0].name
        self.output_name_score = self.model.get_outputs()[1].name  # 'output'
        print(f'init Onnx Model Done!')
    
    def preprocess(self,src_img,bboxes):
        # print("bboxes:",bboxes)
        label,xmin, ymin, xmax, ymax,score = bboxes
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin,self.net_input_w/self.net_input_h)
        scale = scale * 1.0

        inp_h, inp_w = (self.net_input_h,self.net_input_w)

        trans_src_dst,trans_dst_src = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        # cv2.imwrite("./src_img.jpg",src_img)
        # print("trans:",trans_src_dst) #[2,3]
        self.trans_src_dst = trans_src_dst
        self.trans_dst_src = trans_dst_src

        
        self.center = center
        self.scale = scale
        # exit()
        img = cv2.warpAffine(src_img, trans_src_dst, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        self.crop_im = img
        input_image = img.copy()
        # cv2.imwrite("./aaa.jpg",img)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)
        input_tensor = img.unsqueeze(0)
        return input_tensor.numpy() #, bbox ,input_image
    
    def inference(self,input_tensor):
        pose_coords,pose_scores = self.model.run([self.output_name_xy,self.output_name_score], {self.input_name: input_tensor})
        #  = self.model(input_tensor)
        return pose_coords,pose_scores

    def get_results(self,image,bbox):
        input_tensor = self.preprocess(image,bbox)
        pose_coords,pose_scores = self.inference(input_tensor)
        # print("pose_coords.shape:",pose_coords)#(1, 21, 2)

        preds = np.zeros_like(pose_coords)
        for i in range(pose_coords.shape[0]):
            for j in range(pose_coords.shape[1]):
                pose_coords[i,j,0] = (pose_coords[i,j,0]+0.5)*224
                pose_coords[i,j,1] = (pose_coords[i,j,1]+0.5)*224
                preds[i, j, 0:2] = affine_transform(pose_coords[i, j, 0:2],self.trans_dst_src)
        return preds,pose_scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        # if cfg.model.arch.backbone.name == "RepVGG":
        #     deploy_config = cfg.model
        #     deploy_config.arch.backbone.update({"deploy": True})
        #     deploy_model = build_model(deploy_config)
        #     from nanodet.model.backbone.repvgg import repvgg_det_model_convert
        #
        #     model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        print("results:",results)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=True)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()

    handpose = HandPose(model_path="./weights/hand_points.onnx")

    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            print("meta:",meta['raw_img'][0].shape)
            ori_image = meta["raw_img"][0].copy()
            
            score_thresh = 0.25######0.35-----0.25
            result_image = predictor.visualize(res[0], meta, cfg.class_names, score_thresh)

            # 手部关键点检测
            dets = res[0]
            all_box = []
            if len(dets[0]) == 0 and len(dets[1]) == 0:
                continue
            
            for label in dets:
                for bbox in dets[label]:
                    score = bbox[-1]
                    if score > score_thresh:
                        x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                        all_box.append([label, x0, y0, x1, y1, score])
            if len(all_box) == 0:
                continue
            # print("all_box:",all_box)
            for bbox in all_box:
                pose_coords,pose_scores = handpose.get_results(ori_image,bbox)
                for i in range(21):
                    # print(pose_coords[0,i,0])
                    x = int(pose_coords[0,i,0])
                    y = int(pose_coords[0,i,1])
                    cv2.circle(result_image,(x,y),3,(0,0,255),-1)
                for item,color in zip(joint_pairs,joint_colors):
                    # print("item:",item)
                    x1 = int(pose_coords[0,item[0],0])
                    y1 = int(pose_coords[0,item[0],1])
                    x2 = int(pose_coords[0,item[1],0])
                    y2 = int(pose_coords[0,item[1],1])
                    cv2.line(result_image,(x1,y1),(x2,y2),(0,255,0),1)
                    cv2.imshow('handpose_detection', result_image)
            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                print("save_file_name:",save_file_name)
                cv2.imwrite(save_file_name, result_image)

            ch = cv2.waitKey(0)

            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

############视频和摄像###############################################
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, args.path.replace("\\", "/").split("/")[-1])
            if args.demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        while True:
            ret_val, frame = cap.read()
            #cv2.imshow("video",frame)
            if ret_val:
                meta, res = predictor.inference(frame)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)

                ori_image = meta["raw_img"][0].copy()

                score_thresh = 0.25######0.35-----0.25
                result_image = predictor.visualize(res[0], meta, cfg.class_names, score_thresh)

                # 手部关键点检测
                dets = res[0]
                all_box = []
                if len(dets[0]) > 0 or len(dets[1]) > 0:
                    for label in dets:
                        for bbox in dets[label]:
                            score = bbox[-1]
                            if score > score_thresh:
                                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                                all_box.append([label, x0, y0, x1, y1, score])
                    
                    if len(all_box) > 0:
                        for bbox in all_box:
                            pose_coords,pose_scores = handpose.get_results(ori_image,bbox)
                            for i in range(21):
                                # print(pose_coords[0,i,0])
                                x = int(pose_coords[0,i,0])
                                y = int(pose_coords[0,i,1])
                                cv2.circle(result_image,(x,y),3,(0,0,255),-1)
                            for item,color in zip(joint_pairs,joint_colors):
                                # print("item:",item)
                                x1 = int(pose_coords[0,item[0],0])
                                y1 = int(pose_coords[0,item[0],1])
                                x2 = int(pose_coords[0,item[1],0])
                                y2 = int(pose_coords[0,item[1],1])
                                cv2.line(result_image,(x1,y1),(x2,y2),(0,255,0),1)
                                cv2.imshow('handpose_detection', result_image)

                if args.save_result:
                    vid_writer.write(result_image)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break

        # cap.release()


if __name__ == "__main__":
    main()
