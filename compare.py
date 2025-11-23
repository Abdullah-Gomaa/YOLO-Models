"""
YOLO Model Comparison Script
Compare YOLOv5 and YOLOv8 models on the same test images with comprehensive metrics
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import json
from collections import defaultdict
import seaborn as sns


class ModelEvaluator:
    """Handles model loading and inference"""
    
    def __init__(self, model_path, model_type):
        self.model_type = model_type
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        if self.model_type == "yolov5":
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        elif self.model_type == "yolov8":
            model = YOLO(self.model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return model
    
    def predict(self, image_path):
        """Run inference and return results"""
        start_time = time.time()
        
        if self.model_type == "yolov5":
            results = self.model(image_path)
            inference_time = time.time() - start_time
            detections = results.pandas().xyxy[0]
            num_detections = len(detections)
            confidence_scores = detections['confidence'].tolist() if num_detections > 0 else []
            boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist() if num_detections > 0 else []
            classes = detections['class'].tolist() if num_detections > 0 else []
            class_names = detections['name'].tolist() if num_detections > 0 else []
            
        elif self.model_type == "yolov8":
            results = self.model(image_path, verbose=False)
            inference_time = time.time() - start_time
            result = results[0]
            boxes_obj = result.boxes
            num_detections = len(boxes_obj)
            confidence_scores = boxes_obj.conf.cpu().numpy().tolist() if num_detections > 0 else []
            boxes = boxes_obj.xyxy.cpu().numpy().tolist() if num_detections > 0 else []
            classes = boxes_obj.cls.cpu().numpy().astype(int).tolist() if num_detections > 0 else []
            class_names = [result.names[int(c)] for c in classes] if num_detections > 0 else []
        
        return {
            'inference_time': inference_time,
            'num_detections': num_detections,
            'confidence_scores': confidence_scores,
            'boxes': boxes,
            'classes': classes,
            'class_names': class_names,
            'results': results
        }


class GroundTruthLoader:
    """Load ground truth annotations (supports YOLO format)"""
    
    @staticmethod
    def load_yolo_annotation(label_path, img_width, img_height, class_names):
        """
        Load YOLO format annotation (class x_center y_center width height)
        Returns list of ground truth boxes in [xmin, ymin, xmax, ymax, class_id, class_name] format
        """
        if not Path(label_path).exists():
            return []
        
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    xmin = x_center - width / 2
                    ymin = y_center - height / 2
                    xmax = x_center + width / 2
                    ymax = y_center + height / 2
                    
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    gt_boxes.append([xmin, ymin, xmax, ymax, class_id, class_name])
        
        return gt_boxes


class DetectionMetrics:
    """Calculate detection metrics including mAP, precision, recall, F1, confusion matrix"""
    
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_predictions = []
        self.all_ground_truths = []
        self.tp_per_class = defaultdict(int)
        self.fp_per_class = defaultdict(int)
        self.fn_per_class = defaultdict(int)
        self.confusion_matrix_data = defaultdict(lambda: defaultdict(int))
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes [xmin, ymin, xmax, ymax]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, predictions, ground_truths):
        """
        Update metrics with predictions and ground truths for one image
        
        Args:
            predictions: list of [xmin, ymin, xmax, ymax, class_id, class_name, confidence]
            ground_truths: list of [xmin, ymin, xmax, ymax, class_id, class_name]
        """
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truths)
        
        # Match predictions to ground truths
        gt_matched = [False] * len(ground_truths)
        
        # Sort predictions by confidence (descending)
        predictions_sorted = sorted(predictions, key=lambda x: x[6], reverse=True)
        
        for pred in predictions_sorted:
            pred_box = pred[:4]
            pred_class = pred[4]
            pred_class_name = pred[5]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue
                
                gt_box = gt[:4]
                gt_class = gt[4]
                
                if pred_class == gt_class:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                # True Positive
                self.tp_per_class[pred_class_name] += 1
                gt_matched[best_gt_idx] = True
                # Confusion matrix: correct prediction
                self.confusion_matrix_data[pred_class_name][pred_class_name] += 1
            else:
                # False Positive
                self.fp_per_class[pred_class_name] += 1
                # Confusion matrix: prediction without matching GT
                if best_gt_idx >= 0 and best_iou > 0:
                    gt_class_name = ground_truths[best_gt_idx][5]
                    self.confusion_matrix_data[gt_class_name][pred_class_name] += 1
        
        # Count False Negatives (unmatched ground truths)
        for gt_idx, gt in enumerate(ground_truths):
            if not gt_matched[gt_idx]:
                gt_class_name = gt[5]
                self.fn_per_class[gt_class_name] += 1
    
    def calculate_precision_recall_f1(self):
        """Calculate precision, recall, and F1 score per class"""
        metrics = {}
        all_classes = set(list(self.tp_per_class.keys()) + 
                         list(self.fp_per_class.keys()) + 
                         list(self.fn_per_class.keys()))
        
        for class_name in all_classes:
            tp = self.tp_per_class[class_name]
            fp = self.fp_per_class[class_name]
            fn = self.fn_per_class[class_name]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Calculate macro averages
        if metrics:
            avg_precision = np.mean([m['precision'] for m in metrics.values()])
            avg_recall = np.mean([m['recall'] for m in metrics.values()])
            avg_f1 = np.mean([m['f1'] for m in metrics.values()])
        else:
            avg_precision = avg_recall = avg_f1 = 0
        
        return metrics, {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }
    
    def calculate_map(self, iou_thresholds=[0.5]):
        """Calculate mAP at specified IoU thresholds"""
        if not self.all_predictions or not self.all_ground_truths:
            return 0
        
        # Group by class
        predictions_by_class = defaultdict(list)
        ground_truths_by_class = defaultdict(list)
        
        for pred in self.all_predictions:
            predictions_by_class[pred[5]].append(pred)
        
        for gt in self.all_ground_truths:
            ground_truths_by_class[gt[5]].append(gt)
        
        all_classes = set(list(predictions_by_class.keys()) + list(ground_truths_by_class.keys()))
        
        aps = []
        for class_name in all_classes:
            preds = predictions_by_class.get(class_name, [])
            gts = ground_truths_by_class.get(class_name, [])
            
            if not gts:
                continue
            
            # Sort predictions by confidence
            preds = sorted(preds, key=lambda x: x[6], reverse=True)
            
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            
            gt_matched = [False] * len(gts)
            
            for pred_idx, pred in enumerate(preds):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    iou = self.calculate_iou(pred[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / len(gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Calculate AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11
            
            aps.append(ap)
        
        return np.mean(aps) if aps else 0
    
    def get_confusion_matrix(self):
        """Get confusion matrix as numpy array"""
        if not self.confusion_matrix_data:
            return np.array([]), []
        
        classes = sorted(set(list(self.confusion_matrix_data.keys()) + 
                            [k for subdict in self.confusion_matrix_data.values() for k in subdict.keys()]))
        
        matrix = np.zeros((len(classes), len(classes)))
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i][j] = self.confusion_matrix_data[true_class].get(pred_class, 0)
        
        return matrix, classes


class ResultsVisualizer:
    """Handles visualization of results"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_detection_image(self, image_path, yolov5_results, yolov8_results, filename):
        """Save side-by-side comparison of detections"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # YOLOv5 results
        if yolov5_results['results'] is not None:
            yolov5_img = np.squeeze(yolov5_results['results'].render())
            axes[0].imshow(yolov5_img)
        else:
            axes[0].imshow(img)
        axes[0].set_title(f"YOLOv5\nDetections: {yolov5_results['num_detections']}\n"
                         f"Time: {yolov5_results['inference_time']:.3f}s")
        axes[0].axis('off')
        
        # YOLOv8 results
        if yolov8_results['results'] is not None:
            yolov8_img = yolov8_results['results'][0].plot()
            yolov8_img = cv2.cvtColor(yolov8_img, cv2.COLOR_BGR2RGB)
            axes[1].imshow(yolov8_img)
        else:
            axes[1].imshow(img)
        axes[1].set_title(f"YOLOv8\nDetections: {yolov8_results['num_detections']}\n"
                         f"Time: {yolov8_results['inference_time']:.3f}s")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, matrix, classes, model_name):
        """Plot confusion matrix"""
        if matrix.size == 0:
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name.lower()}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_f1(self, yolov5_metrics, yolov8_metrics):
        """Plot precision, recall, F1 comparison"""
        metrics_names = ['Precision', 'Recall', 'F1 Score']
        yolov5_values = [yolov5_metrics['precision'], 
                        yolov5_metrics['recall'], 
                        yolov5_metrics['f1']]
        yolov8_values = [yolov8_metrics['precision'], 
                        yolov8_metrics['recall'], 
                        yolov8_metrics['f1']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, yolov5_values, width, label='YOLOv5', color='#1f77b4')
        bars2 = ax.bar(x + width/2, yolov8_values, width, label='YOLOv8', color='#ff7f0e')
        
        ax.set_ylabel('Score')
        ax.set_title('Detection Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_f1.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_plots(self, metrics):
        """Create comparison plots for all metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Inference time comparison
        axes[0, 0].bar(['YOLOv5', 'YOLOv8'], 
                       [metrics['yolov5']['avg_inference_time'], 
                        metrics['yolov8']['avg_inference_time']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Average Inference Time')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Total detections
        axes[0, 1].bar(['YOLOv5', 'YOLOv8'], 
                       [metrics['yolov5']['total_detections'], 
                        metrics['yolov8']['total_detections']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Total Detections')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Average confidence
        axes[0, 2].bar(['YOLOv5', 'YOLOv8'], 
                       [metrics['yolov5']['avg_confidence'], 
                        metrics['yolov8']['avg_confidence']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[0, 2].set_ylabel('Confidence')
        axes[0, 2].set_title('Average Confidence Score')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # mAP@0.5
        axes[1, 0].bar(['YOLOv5', 'YOLOv8'], 
                       [metrics['yolov5'].get('map50', 0), 
                        metrics['yolov8'].get('map50', 0)],
                       color=['#1f77b4', '#ff7f0e'])
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title('mAP@0.5')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Precision
        axes[1, 1].bar(['YOLOv5', 'YOLOv8'], 
                       [metrics['yolov5'].get('precision', 0), 
                        metrics['yolov8'].get('precision', 0)],
                       color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Recall
        axes[1, 2].bar(['YOLOv5', 'YOLOv8'], 
                       [metrics['yolov5'].get('recall', 0), 
                        metrics['yolov8'].get('recall', 0)],
                       color=['#1f77b4', '#ff7f0e'])
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Recall')
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_summary.png', dpi=150, bbox_inches='tight')
        plt.close()


class MetricsCalculator:
    """Calculate and aggregate metrics"""
    
    @staticmethod
    def calculate_metrics(results_list):
        """Calculate aggregate metrics from all results"""
        total_time = sum(r['inference_time'] for r in results_list)
        total_detections = sum(r['num_detections'] for r in results_list)
        all_confidences = [conf for r in results_list for conf in r['confidence_scores']]
        
        return {
            'total_inference_time': total_time,
            'avg_inference_time': total_time / len(results_list) if results_list else 0,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(results_list) if results_list else 0,
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'min_confidence': np.min(all_confidences) if all_confidences else 0,
            'max_confidence': np.max(all_confidences) if all_confidences else 0,
        }
    
    @staticmethod
    def print_summary(yolov5_metrics, yolov8_metrics, num_images):
        """Print formatted summary"""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"Total images processed: {num_images}")
        print("\n" + "-"*80)
        print(f"{'Metric':<35} {'YOLOv5':>20} {'YOLOv8':>20}")
        print("-"*80)
        
        # Performance metrics
        print(f"{'Avg Inference Time (s)':<35} {yolov5_metrics['avg_inference_time']:>20.4f} {yolov8_metrics['avg_inference_time']:>20.4f}")
        print(f"{'Total Detections':<35} {yolov5_metrics['total_detections']:>20d} {yolov8_metrics['total_detections']:>20d}")
        print(f"{'Avg Detections/Image':<35} {yolov5_metrics['avg_detections_per_image']:>20.2f} {yolov8_metrics['avg_detections_per_image']:>20.2f}")
        print(f"{'Avg Confidence':<35} {yolov5_metrics['avg_confidence']:>20.4f} {yolov8_metrics['avg_confidence']:>20.4f}")
        
        # Detection metrics (if available)
        if 'map50' in yolov5_metrics:
            print("\n" + "-"*80)
            print(f"{'mAP@0.5':<35} {yolov5_metrics['map50']:>20.4f} {yolov8_metrics['map50']:>20.4f}")
            print(f"{'Precision':<35} {yolov5_metrics['precision']:>20.4f} {yolov8_metrics['precision']:>20.4f}")
            print(f"{'Recall':<35} {yolov5_metrics['recall']:>20.4f} {yolov8_metrics['recall']:>20.4f}")
            print(f"{'F1 Score':<35} {yolov5_metrics['f1']:>20.4f} {yolov8_metrics['f1']:>20.4f}")
        
        print("="*80 + "\n")


def main(yolov5_path, yolov8_path, test_images_dir, labels_dir=None, 
         class_names=None, output_dir="results", iou_threshold=0.5):
    """
    Main comparison function
    
    Args:
        yolov5_path: Path to YOLOv5 model weights
        yolov8_path: Path to YOLOv8 model weights
        test_images_dir: Directory containing test images
        labels_dir: Directory containing ground truth labels (YOLO format)
        class_names: Dict mapping class IDs to names {0: 'person', 1: 'car', ...}
        output_dir: Directory to save results
        iou_threshold: IoU threshold for matching predictions to ground truths
    """
    print("Loading models...")
    yolov5_eval = ModelEvaluator(yolov5_path, "yolov5")
    yolov8_eval = ModelEvaluator(yolov8_path, "yolov8")
    
    # Get test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    test_images = [f for f in Path(test_images_dir).iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not test_images:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Initialize components
    visualizer = ResultsVisualizer(output_dir)
    gt_loader = GroundTruthLoader()
    
    yolov5_results = []
    yolov8_results = []
    
    # Initialize detection metrics
    yolov5_det_metrics = DetectionMetrics(iou_threshold)
    yolov8_det_metrics = DetectionMetrics(iou_threshold)
    
    has_ground_truth = labels_dir is not None and Path(labels_dir).exists()
    
    # Process each image
    print("\nProcessing images...")
    for idx, img_path in enumerate(test_images, 1):
        print(f"  [{idx}/{len(test_images)}] {img_path.name}")
        
        # Run inference
        yolov5_result = yolov5_eval.predict(str(img_path))
        yolov8_result = yolov8_eval.predict(str(img_path))
        
        yolov5_results.append(yolov5_result)
        yolov8_results.append(yolov8_result)
        
        # Load ground truth and update metrics
        if has_ground_truth and class_names:
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            ground_truths = gt_loader.load_yolo_annotation(
                label_path, img_width, img_height, class_names
            )
            
            # Format predictions for metrics calculation
            yolov5_preds = [[*box, cls, cls_name, conf] 
                           for box, cls, cls_name, conf 
                           in zip(yolov5_result['boxes'], 
                                 yolov5_result['classes'],
                                 yolov5_result['class_names'],
                                 yolov5_result['confidence_scores'])]
            
            yolov8_preds = [[*box, cls, cls_name, conf] 
                           for box, cls, cls_name, conf 
                           in zip(yolov8_result['boxes'], 
                                 yolov8_result['classes'],
                                 yolov8_result['class_names'],
                                 yolov8_result['confidence_scores'])]
            
            yolov5_det_metrics.update(yolov5_preds, ground_truths)
            yolov8_det_metrics.update(yolov8_preds, ground_truths)
        
        # Save visualization
        output_filename = f"comparison_{img_path.stem}.png"
        visualizer.save_detection_image(img_path, yolov5_result, yolov8_result, output_filename)
    
    # Calculate basic metrics
    print("\nCalculating metrics...")
    calc = MetricsCalculator()
    yolov5_metrics = calc.calculate_metrics(yolov5_results)
    yolov8_metrics = calc.calculate_metrics(yolov8_results)
    
    # Calculate detection metrics if ground truth available
    if has_ground_truth and class_names:
        print("Calculating detection metrics (precision, recall, F1, mAP)...")
        
        # YOLOv5 metrics
        yolov5_class_metrics, yolov5_avg_metrics = yolov5_det_metrics.calculate_precision_recall_f1()
        yolov5_map = yolov5_det_metrics.calculate_map()
        yolov5_confusion, yolov5_classes = yolov5_det_metrics.get_confusion_matrix()
        
        # YOLOv8 metrics
        yolov8_class_metrics, yolov8_avg_metrics = yolov8_det_metrics.calculate_precision_recall_f1()
        yolov8_map = yolov8_det_metrics.calculate_map()
        yolov8_confusion, yolov8_classes = yolov8_det_metrics.get_confusion_matrix()
        
        # Add to metrics dict
        yolov5_metrics.update({
            'map50': yolov5_map,
            'precision': yolov5_avg_metrics['precision'],
            'recall': yolov5_avg_metrics['recall'],
            'f1': yolov5_avg_metrics['f1'],
            'class_metrics': yolov5_class_metrics
        })
        
        yolov8_metrics.update({
            'map50': yolov8_map,
            'precision': yolov8_avg_metrics['precision'],
            'recall': yolov8_avg_metrics['recall'],
            'f1': yolov8_avg_metrics['f1'],
            'class_metrics': yolov8_class_metrics
        })
        
        # Plot confusion matrices
        print("Creating confusion matrices...")
        visualizer.plot_confusion_matrix(yolov5_confusion, yolov5_classes, "YOLOv5")
        visualizer.plot_confusion_matrix(yolov8_confusion, yolov8_classes, "YOLOv8")
        
        # Plot precision/recall/F1
        visualizer.plot_precision_recall_f1(yolov5_avg_metrics, yolov8_avg_metrics)
        
        # Save per-class metrics to JSON
        metrics_report = {
            'yolov5': {
                'overall': {
                    'mAP@0.5': float(yolov5_map),
                    'precision': float(yolov5_avg_metrics['precision']),
                    'recall': float(yolov5_avg_metrics['recall']),
                    'f1_score': float(yolov5_avg_metrics['f1'])
                },
                'per_class': {k: {metric: float(v) for metric, v in cls_metrics.items()} 
                             for k, cls_metrics in yolov5_class_metrics.items()}
            },
            'yolov8': {
                'overall': {
                    'mAP@0.5': float(yolov8_map),
                    'precision': float(yolov8_avg_metrics['precision']),
                    'recall': float(yolov8_avg_metrics['recall']),
                    'f1_score': float(yolov8_avg_metrics['f1'])
                },
                'per_class': {k: {metric: float(v) for metric, v in cls_metrics.items()} 
                             for k, cls_metrics in yolov8_class_metrics.items()}
            }
        }
        
        with open(Path(output_dir) / 'metrics_report.json', 'w') as f:
            json.dump(metrics_report, f, indent=2)
        
        print(f"Detailed metrics saved to: {output_dir}/metrics_report.json")
    
    # Create summary visualization
    visualizer.create_summary_plots({
        'yolov5': yolov5_metrics,
        'yolov8': yolov8_metrics
    })
    
    # Print summary
    calc.print_summary(yolov5_metrics, yolov8_metrics, len(test_images))
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Individual comparisons: comparison_*.png")
    print(f"  - Summary plot: comparison_summary.png")
    if has_ground_truth and class_names:
        print(f"  - Confusion matrices: confusion_matrix_yolov5.png, confusion_matrix_yolov8.png")
        print(f"  - Precision/Recall/F1: precision_recall_f1.png")
        print(f"  - Detailed metrics: metrics_report.json")


if __name__ == "__main__":
    # Example usage
    YOLOV5_MODEL = "./yolov5/runs/train/MyYoloV5Model/weights/best.pt"  # or path to your custom model
    YOLOV8_MODEL = "./runs/detect/MyYoloV8Models2/weights/best.pt"  # or path to your custom model
    TEST_IMAGES_DIR = "./Safety-Vests-14/test/images"
    LABELS_DIR = "./Safety-Vests-14/test/labels"  # Optional: directory with YOLO format labels
    OUTPUT_DIR = "./comparison_results"
    
    # Optional: Define class names for your dataset
    # If None, metrics requiring ground truth will be skipped
    CLASS_NAMES = {
        0: 'no_safety_vest',
        1: 'safety_vest',
    }
    
    main(
        yolov5_path=YOLOV5_MODEL,
        yolov8_path=YOLOV8_MODEL,
        test_images_dir=TEST_IMAGES_DIR,
        labels_dir=LABELS_DIR,  # Set to None if no ground truth available
        class_names=CLASS_NAMES,  # Set to None if no ground truth available
        output_dir=OUTPUT_DIR,
        iou_threshold=0.5
    )