"""
Gap Junction Entity Detection Module
Detects and evaluates individual gap junction entities using 3D connected components.
Tommy Tang & Kirpa Chandok
October 2025
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
import cc3d  # Using cc3d for connected components
from rtree import index
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##parallel processing 
import multiprocessing as mp


class GapJunctionEntityDetector2DOptimized:
    def __init__(self, threshold=0.5, iou_threshold=0.001, connectivity=8, min_size=25, n_workers = 4):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.connectivity = connectivity  # 6, 18, or 26 for 3D; 4 or 8 for 2D
        self.min_size = min_size  # Minimum size to consider a 3d component valid
        self.n_workers = n_workers if n_workers > 0 else max(1, mp.cpu_count() - 1)

    
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
    
    def process_entities_parallel(self, label_ids, labelled_img, min_size):
        results = []
        for lab_id in label_ids:
            positions = np.argwhere(labelled_img == lab_id)
            if positions.shape[0] < min_size:
                continue
            
            pixel_set = set((int(r), int(c)) for r, c in positions)
            min_row, min_col = positions.min(axis=0)
            max_row, max_col = positions.max(axis=0)
            
            bbox = (int(min_col), int(min_row), int(max_col), int(max_row))
            
            results.append((lab_id, pixel_set, bbox))
    
        return results
    
    def extract_entities_2d(self, img, build_index=False):
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

        ## Prepare batches for parallel processing
        label_ids = list(range(1, num_raw + 1))
        batch_size = max(1, len(label_ids) // self.n_workers)

        batches = []
        for i in range(0, len(label_ids), batch_size):
            batch_labels = label_ids[i: i + batch_size]
            batches.append(batch_labels)

        entity_pixels = {}
        entity_bboxes = {}

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Create a partial function with the fixed arguments
            process_func = partial(
                self.process_entities_parallel,
                labelled_img=labelled_img,
                min_size=self.min_size
            )
            
            # Submit all batches - only pass the varying argument (batch)
            futures = []
            for batch in batches:
                future = executor.submit(process_func, batch)  # batch becomes label_ids
                futures.append(future)
            
            # Collect results
            next_id = 0
            for future in as_completed(futures):
                batch_results = future.result()
                for lab_id, pixels, bbox in batch_results:
                    entity_pixels[next_id] = pixels
                    entity_bboxes[next_id] = bbox
                    next_id += 1
       
        # optinally build R-tree index -- set to false default
        idx = None
        if build_index:
            idx = index.Index()
            for eid, bbox in entity_bboxes.items():
                idx.insert(eid, bbox)
        
        num_entities = len(entity_pixels)
        
        return labelled_img, idx, num_entities, entity_pixels, entity_bboxes
    
    def _iou_sets(self, set1, set2):
        if not set1 and not set2:
            return 0.0
        inter = len(set1 & set2)
        union = len(set1 | set2)
        return inter / union if union else 0.0
    
    def entity_metrics_2d_parallel(self, pred_img, gt_img):
        """
        Returns:
            shared_positions: list[(row, col)]
            metrics_dict: {tp, fp, fn, precision, recall, f1_score}
            shared_entities: int
        """
        _, pred_idx, pred_n, pred_pixels, pred_bboxes = self.extract_entities_2d(pred_img)
        _, gt_idx, gt_n, gt_pixels, gt_bboxes = self.extract_entities_2d(gt_img)
        
        # If no predictions or GT, return zeros
        if pred_n == 0 or gt_n == 0:
            metrics_dict = {
                'tp': 0,
                'fp': pred_n,
                'fn': gt_n,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            return [], metrics_dict, 0
        
        # Compute all IOUs in parallel
        pred_ids = list(pred_bboxes.keys())
        batch_size = max(1, len(pred_ids) // self.n_workers)
        batches = []
        for i in range(0, len(pred_ids), batch_size):
            batch = pred_ids[i:i + batch_size]
            batches.append(batch)
        
        # Store all candidate matches: (pred_id, gt_id, iou, inter_pixels)
        all_candidates = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Pass gt_bboxes dict - workers will rebuild the spatial index
            process_func = partial(
                self._compute_candidates_parallel,
                pred_pixels=pred_pixels,
                pred_bboxes=pred_bboxes,
                gt_bboxes=gt_bboxes,  # Pass dict, not index
                gt_pixels=gt_pixels,
                iou_threshold=self.iou_threshold
            )
            
            futures = []
            for batch in batches:
                future = executor.submit(process_func, batch)
                futures.append(future)
            
            for future in as_completed(futures):
                batch_candidates = future.result()
                all_candidates.extend(batch_candidates)
        
        # Sort by IOU descending and perform greedy matching sequentially
        all_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by IOU
        
        matched_pred = set()
        matched_gt = set()
        shared_positions = []
        
        for pred_id, gt_id, iou, inter_pixels in all_candidates:
            if pred_id not in matched_pred and gt_id not in matched_gt:
                matched_pred.add(pred_id)
                matched_gt.add(gt_id)
                shared_positions.extend(inter_pixels)
        
        tp = len(matched_pred)
        fp = pred_n - tp
        fn = gt_n - tp
        
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        
        metrics_dict = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        shared_entities = tp
        return shared_positions, metrics_dict, shared_entities

    def _compute_candidates_parallel(self, pred_ids, pred_pixels, pred_bboxes, gt_bboxes, gt_pixels, iou_threshold):
        """Compute all candidate matches above threshold for a batch of predictions.
        """
        candidates = []

        ##rebuild spatial index
        gt_idx = index.Index()
        for gt_id, bbox in gt_bboxes.items():
            gt_idx.insert(gt_id, bbox)
        
        for pred_id in pred_ids:
            pb = pred_bboxes[pred_id]
            
            # Use spatial index to find GT candidates
            gt_candidates = [item.id for item in gt_idx.intersection(pb, objects=True)]
            
            for gt_id in gt_candidates:
                iou = self._iou_sets(pred_pixels[pred_id], gt_pixels[gt_id])
                if iou >= iou_threshold:
                    inter_pixels = pred_pixels[pred_id] & gt_pixels[gt_id]
                    candidates.append((pred_id, gt_id, iou, inter_pixels))
    
        return candidates
    
   
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