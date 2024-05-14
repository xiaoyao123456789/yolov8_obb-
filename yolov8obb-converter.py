import os
import random
import shutil
import xml.etree.ElementTree as ET
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class_mapping = {
    "Abandoned": 0,
    "SignLine": 1
}

def process_xml(xml_file):
    tree = ET.parse(xml_file)
    name = os.path.splitext(os.path.basename(xml_file))[0]
    objs_info = []

    objs = tree.findall('object')
    for obj in objs:
        cls = obj.find('name').text
        # if cls not in cls_list:
        #     continue

        if obj.find('robndbox') is None:
            box = obj.find('bndbox')
            if box is None:
                continue
            x0 = max(int(float(box.find('xmin').text)), 0)
            y0 = max(int(float(box.find('ymin').text)), 0)
            x1 = max(int(float(box.find('xmax').text)), 0)
            y1 = max(int(float(box.find('ymax').text)), 0)
            x2, y2, x3, y3 = x1, y0, x0, y1
        else:
            box = obj.find('robndbox')
            if box is None:
                continue
            cx = float(box.find('cx').text)
            cy = float(box.find('cy').text)
            w = float(box.find('w').text)
            h = float(box.find('h').text)
            angle = float(box.find('angle').text)
            cosA = math.cos(math.radians(angle))
            sinA = math.sin(math.radians(angle))
            x0 = int(cx - w / 2 * cosA - h / 2 * sinA)
            y0 = int(cy - w / 2 * sinA + h / 2 * cosA)
            x1 = int(cx + w / 2 * cosA - h / 2 * sinA)
            y1 = int(cy + w / 2 * sinA + h / 2 * cosA)
            x2 = int(cx + w / 2 * cosA + h / 2 * sinA)
            y2 = int(cy + w / 2 * sinA - h / 2 * cosA)
            x3 = int(cx - w / 2 * cosA + h / 2 * sinA)
            y3 = int(cy - w / 2 * sinA - h / 2 * cosA)

        objs_info.append((x0, y0, x1, y1, x2, y2, x3, y3, cls))

    return name, objs_info

def convert_label(image_name, image_width, image_height, objs_info, save_dir):
    """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""

    save_path = os.path.join(save_dir, f"{image_name}.txt")

    with open(save_path, "w") as g:
        if not objs_info:
            return

        for obj_info in objs_info:
            class_name = obj_info[8]
            class_idx = class_mapping.get(class_name)
            if class_idx is None:
                continue
            coords = obj_info[:8]
            normalized_coords = [
                coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
            ]
            formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
            g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

def preprocess_data(data_dir, split_ratio=0.7):
    # 创建训练和测试集目录
    images_train_dir = os.path.join(data_dir, 'images', 'train')
    images_val_dir = os.path.join(data_dir, 'images', 'val')
    labels_train_dir = os.path.join(data_dir, 'labels', 'train')
    labels_val_dir = os.path.join(data_dir, 'labels', 'val')
    xml_train_dir = os.path.join(data_dir, 'xml', 'train')
    xml_val_dir = os.path.join(data_dir, 'xml', 'val')
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    os.makedirs(xml_train_dir, exist_ok=True)
    os.makedirs(xml_val_dir, exist_ok=True)

    roxml_path = data_dir
    xml_files = [os.path.join(roxml_path, file) for file in os.listdir(roxml_path) if file.endswith('.xml')]
    total_files = len(xml_files)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_xml, xml_files))

    for result in tqdm(results, total=total_files, desc="Generating txt files"):
        image_name, objs_info = result
        convert_label(image_name, 3000, 3000, objs_info, roxml_path)

    # 划分数据集并移动文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    random.shuffle(image_files)
    train_size = int(len(image_files) * split_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    for filename in train_files:
        img_path = os.path.join(data_dir, filename)
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        shutil.move(img_path, os.path.join(images_train_dir, filename))
        if not os.path.exists(txt_path):
            open(os.path.join(labels_train_dir, os.path.splitext(filename)[0] + '.txt'), 'a').close()
        else:
            shutil.move(txt_path, os.path.join(labels_train_dir, os.path.splitext(filename)[0] + '.txt'))
        shutil.move(os.path.join(roxml_path, os.path.splitext(filename)[0] + '.xml'), os.path.join(xml_train_dir, os.path.splitext(filename)[0] + '.xml'))
    for filename in val_files:
        img_path = os.path.join(data_dir, filename)
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        shutil.move(img_path, os.path.join(images_val_dir, filename))
        if not os.path.exists(txt_path):
            open(os.path.join(labels_val_dir, os.path.splitext(filename)[0] + '.txt'), 'a').close()
        else:
            shutil.move(txt_path, os.path.join(labels_val_dir, os.path.splitext(filename)[0] + '.txt'))
        # 移动XML文件到相应目录
        shutil.move(os.path.join(roxml_path, os.path.splitext(filename)[0] + '.xml'), os.path.join(xml_val_dir, os.path.splitext(filename)[0] + '.xml'))
if __name__ == '__main__':
    data_dir = r'E:\Test_xml\test_data'  # 数据集目录，包含图片和对应的xml文件
    split_ratio = 0.9  # 划分比例
    preprocess_data(data_dir, split_ratio)
