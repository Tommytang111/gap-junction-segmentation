"""
Gap Junction Entity Detection Module
Detects and evaluates individual gap junction entities using 3D connected components.
Tommy Tang & Kirpa Chandok
October 2025
"""

import numpy as np
import cc3d  # Using cc3d for connected components
from rtree import index
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GapJunctionEntityDetector2D:
    def __init__(self, threshold=0.5, iou_threshold=0.001, connectivity=8, min_size=25):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.connectivity = connectivity  # 6, 18, or 26 for 3D; 4 or 8 for 2D
        self.min_size = min_size  # Minimum size to consider a 3d component valid

    def _coerce_bounds(self, box_like):
        """Return (minx, miny, maxx, maxy) from:
        - (minx, miny, maxx, maxy)
        - ((minx, miny), (maxx, maxy))
        - object with .bounds
        """
        if hasattr(box_like, "bounds"):
            return box_like.bounds
        if isinstance(box_like, (tuple, list)):
            if len(box_like) == 4:
                return tuple(box_like)
            if len(box_like) == 2 and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in box_like):
                (minx, miny), (maxx, maxy) = box_like
                return (minx, miny, maxx, maxy)
        raise TypeError(f"Unrecognized bbox format: {type(box_like)} {box_like}")

    def _item_bounds(self, item):
        """Prefer the stored object if present; otherwise use the Item’s bounds."""
        obj = getattr(item, "object", None)
        if obj is not None:
            try:
                return self._coerce_bounds(obj)
            except TypeError:
                pass
        return item.bounds  # (minx, miny, maxx, maxy)

    
    def extract_entities_2d(self, img):
        """
        Extract connected component entities from a 2D image.

        Returns:
            labelled_img: HxW label map (0 = background)
            idx:          R-tree index over entity boxes (x=col, y=row)
            num_entities: number of kept entities (after min_size filtering)
            entity_pixels: dict[int -> set[(row, col)]]
            entity_bboxes: dict[int -> (minx, miny, maxx, maxy)]  # x=col, y=row
        """
        if img is None:
            raise ValueError("Input image is None")

        # Convert to binary
        if img.dtype in (np.float32, np.float64):
            binary_img = (img >= self.threshold).astype(np.uint8)
        elif img.dtype == np.uint8:
            binary_img = (img >= 127).astype(np.uint8) if img.max() > 1 else img.astype(np.uint8)
        else:
            binary_img = (img > 0).astype(np.uint8)

        labelled_img, num_raw = cc3d.connected_components(binary_img, connectivity=8, return_N=True)

        idx = index.Index()
        entity_pixels = {}
        entity_bboxes = {}

        next_id = 0  # 0-based ids for our own dictionaries
        for lab_id in range(1, num_raw + 1):
            rows, cols = np.where(labelled_img == lab_id)
            if rows.size == 0:
                continue

            # Optional min_size filter
            if rows.size < self.min_size:
                continue

            # x = col, y = row (R-tree expects (minx, miny, maxx, maxy))
            minx, maxx = int(cols.min()), int(cols.max())
            miny, maxy = int(rows.min()), int(rows.max())

            # Save pixels and bbox
            pixels = set(zip(rows.tolist(), cols.tolist()))
            entity_pixels[next_id] = pixels
            entity_bboxes[next_id] = (minx, miny, maxx, maxy)

            # Index the bbox
            idx.insert(next_id, (minx, miny, maxx, maxy))
            next_id += 1

        num_entities = next_id
        return labelled_img, idx, num_entities, entity_pixels, entity_bboxes

    
    def _iou_sets(self, set1, set2):
        if not set1 and not set2:
            return 0.0
        inter = len(set1 & set2)
        union = len(set1 | set2)
        return inter / union if union else 0.0

    def entity_metrics_2d(self, pred_img, gt_img):
        """
        Returns:
        shared_positions: list[(row, col)]  # pixel-wise intersections of matched pairs
        metrics_dict: {tp, fp, fn, precision, recall, f1_score}
        shared_entities: int                 # number of matched pairs
        """
        _, pred_idx, pred_n, pred_pixels, pred_bboxes = self.extract_entities_2d(pred_img)
        _, gt_idx, gt_n, gt_pixels, gt_bboxes       = self.extract_entities_2d(gt_img)

        # Optional: visualize boxes
        self.visualize_boxes(pred_idx, title="Predicted Entities")
        self.visualize_boxes(gt_idx,   title="Ground Truth Entities")

        matched_pred = []
        matched_gt   = set()
        shared_positions = []

        # Iterate over predicted entities by id
        for pred_id, pb in pred_bboxes.items():
            # Candidate GT entities overlapping the pred bbox
            candidates = [it.id for it in gt_idx.intersection(pb, objects=True)]

            best_iou = 0.0
            best_gt  = None

            for gt_id in candidates:
                if gt_id in matched_gt:
                    continue
                iou = self._iou_sets(pred_pixels[pred_id], gt_pixels[gt_id])
                if iou > best_iou:
                    best_iou = iou
                    best_gt  = gt_id

            if best_gt is not None and best_iou >= self.iou_threshold:
                matched_pred.append(pred_id)
                matched_gt.add(best_gt)
                # Add intersection pixels for visualization
                inter_pixels = pred_pixels[pred_id] & gt_pixels[best_gt]
                shared_positions.extend(inter_pixels)

        tp = len(matched_pred)
        fp = pred_n - tp
        fn = gt_n - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics_dict = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1
        }
        shared_entities = tp

        return shared_positions, metrics_dict, shared_entities

    

    def visualize_boxes(self, idx, bounds=(-float('inf'), -float('inf'), float('inf'), float('inf')), ax=None, title=None):
        # Collect rectangles as (x, y, w, h)
        rects = []
        for item in idx.intersection(bounds, objects=True):
            minx, miny, maxx, maxy = self._item_bounds(item)
            rects.append((minx, miny, maxx - minx, maxy - miny))

        # Plot
        if ax is None:
            fig, ax = plt.subplots(1)
        for (x, y, w, h) in rects:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1))
        if title:
            ax.set_title(title)
        ax.autoscale()
        ax.set_aspect('equal', adjustable='box')
        plt.show()


            