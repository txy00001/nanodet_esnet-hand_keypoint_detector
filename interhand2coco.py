import json
import os
import numpy as np
YOLO_DARKNET_SUB_DIR = "YOLO_darknet"
import cv2
classes = [
    "left_hand",

    "right_hand",
]

coco_format = {"images": [{}], "categories": [], "annotations": [{}]}
ori_image_path = "~/dataset/nano_keypoints/annotations/val"

def debug(image_path ,bbox,class_names):
    img_file = cv2.imread(image_path)
    # Draw bounding box
    cv2.rectangle(
                img_file,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                (0,0,255),3)

    cv2.putText(img_file,class_names,(int(bbox[0]), int(bbox[1])), 1, 2, (0,255,0), thickness=1)
    save_path = os.path.join("./debug_images",os.path.basename(image_path))
    cv2.imwrite(save_path,img_file)


def get_coco_annotaion(image_item,anno_item):
    """_summary_

    Args:
        image_item (Dict): # {'id': 0, 'file_name': 'Capture0/ROM07_Rt_Finger_Occlusions/cam400262/image23330.jpg', 'width': 334, 'height': 512, 'capture': 0, 'subject': 9, 'seq_name': 'ROM07_Rt_Finger_Occlusions', 'camera': '400262', 'frame_idx': 23330}
        anno_item (Dict): {'id': 0, 'image_id': 0, 'bbox': [42.103736877441406, 139.23715209960938, 268.2574462890625, 191.54367065429688], 'joint_valid': [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], 'hand_type': 'right', 'hand_type_valid': 1}
    """
    annotation_id = anno_item["id"]
    image_id = anno_item["image_id"]
    bbox = anno_item["bbox"]
    area = bbox[2] * bbox[3]
    category_id = 2 if anno_item["hand_type"] == "right" else 1
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": None,
    }
    debug(os.path.join(ori_image_path,image_item['file_name']),bbox,anno_item["hand_type"])
    return image_item,annotation

def main():
    root_dir = "~/dataset/nano_keypoints/"
    phase = "val"
    ann_prefix_path = f"annotations/{phase}/InterHand2.6M_{phase}_data.json"
    images_dir = os.path.join(root_dir,f"images/{phase}")
    
    ann_path = os.path.join(root_dir,ann_prefix_path)
    output_path = ann_path.replace("InterHand2.6M","Coco")
    print("output_path:",output_path)
    with open(ann_path,'r') as f:
        obj = json.load(f)
        assert len(obj["images"]) == len(obj["annotations"]),"{} {} len error".format(len(obj["images"]),len(obj["annotations"]))
        annotations = []
        images_annotations =[]
        count  = 0
        for image_item,anno_item in zip(obj["images"],obj["annotations"]):
            image_ann,annotation = get_coco_annotaion(image_item,anno_item)
            images_annotations.append(image_ann)
            annotations.append(annotation)
            count+=1
            if count % 1000 == 0:
                print("{}/{} have been processed!!!".format(count,len(obj["images"])))
            # 
    coco_format["images"] = images_annotations
    coco_format["annotations"] = annotations
    for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

    print("output_path:",output_path)
    with open(output_path, "w") as outfile:
        json.dump(coco_format, outfile, indent=4)

    print("Finished!")
if __name__ == "__main__":
    main()