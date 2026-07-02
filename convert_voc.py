#!/usr/bin/env python3
"""Convert VOC 2007 XML annotations to YOLO TXT format."""
import xml.etree.ElementTree as ET
import os
from pathlib import Path

# VOC class names (same order as voc.yaml)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
class_to_idx = {name: idx for idx, name in enumerate(VOC_CLASSES)}

def convert_xml_to_yolo(xml_path, img_w, img_h):
    """Convert a single XML annotation to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    if size is not None:
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
    
    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_to_idx:
            continue
        cls_id = class_to_idx[cls_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2
        cy = ymin + h / 2
        
        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    return lines

def main():
    voc_root = Path('/datasets/VOCdevkit/VOC2007')
    annotations_dir = voc_root / 'Annotations'
    labels_dir = voc_root / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"Converting {len(xml_files)} XML annotations...")
    
    for xml_path in xml_files:
        img_id = xml_path.stem
        lines = convert_xml_to_yolo(xml_path, 500, 375)
        if lines:
            label_path = labels_dir / f"{img_id}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines))
    
    print(f"Done! Labels saved to {labels_dir}")

if __name__ == '__main__':
    main()