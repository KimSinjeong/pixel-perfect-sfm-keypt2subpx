from collections import defaultdict, Counter
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .._pixsfm import _keypoint_adjustment as ka
from .. import base, features, logger
from ..util.misc import to_ctr, to_optim_ctr

from tqdm import tqdm

def find_problem_labels(track_labels: List[int], max_per_problem: int,
                        track_edge_counts: Optional[List] = None):
    if track_edge_counts is None:
        track_count = Counter(track_labels)
    else:
        track_count = {i: v for i, v in enumerate(track_edge_counts)}
        track_count = Counter(track_count)
    if max_per_problem == -1:
        max_per_problem = max(track_count.values())
    bins = []
    track_label_to_problem = [-1] * len(track_count)

    # First-fit-decreasing bin-packing
    start = 0
    last_v = sys.maxsize
    for k, v in track_count.most_common():
        if v < last_v:
            start = 0
            last_v = v
        found = False
        if v < max_per_problem:
            for i in range(start, len(bins)):
                if bins[i] + v <= max_per_problem:
                    bins[i] += v
                    track_label_to_problem[k] = i
                    found = True
                    start = i  # Avoid looping over full bins
                    break
        if not found:
            track_label_to_problem[k] = len(bins)
            start = len(bins)  # Avoid looping over full bins
            bins.append(v)
    problem_labels = [track_label_to_problem[v] for v in track_labels]
    bin_arr = np.array(bins)
    num_oversized_problems = np.sum(bin_arr > max_per_problem)
    max_in_problem = np.max(bin_arr)
    if num_oversized_problems > 0 and max_per_problem > -1:
        logger.warning(
            "%d / %d problems have more than %d keypoints.\n" %
            (num_oversized_problems, len(bins), max_per_problem)
            + "         Maximum keypoints in a problem: %d" % max_in_problem
        )
    if -1 in problem_labels:
        raise ValueError
    return problem_labels, bins

def create_meshgrid(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x

class SpatialSoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2

class KeypointAdjuster:
    default_conf = {
        'strategy': 'featuremetric',
        'apply': True,
        'interpolation': base.interpolation_default_conf,
        'level_indices': None,
        'max_kps_per_problem': 50,
        'optimizer': {
            'loss': {
                'name': 'cauchy',
                'params': [0.25]
            },
            'solver': {
                **base.solver_default_conf,
                'parameter_tolerance': 1.0e-5,
                'num_threads': 1
            },
            'print_summary': False,
            'bound': 4.0,
            'num_threads': -1
        },
        'split_in_subproblems': True
    }
    callbacks = []

    @classmethod
    def create(cls, conf):
        strategy_to_solver = {
            "featuremetric": FeatureMetricKeypointAdjuster,
            "topological_reference": TopologicalReferenceKeypointAdjuster,
            "keypt2subpx": Keypt2SubpxKeypointAdjuster
        }
        strategy = conf["strategy"] if "strategy" in conf else \
            cls.default_conf["strategy"]
        return strategy_to_solver[strategy](conf)

    def refine(self,
               keypoints_dict: base.Map_NameKeypoints,
               feature_set: features.FeatureSet,
               graph: base.Graph,
               track_labels: List[int],
               root_labels: List[int],
               problem_setup=None,
               descriptors=None) -> dict:
        return NotImplementedError

    def refine_multilevel(self,
                          keypoints_dict,
                          feature_manager: features.FeatureManager,
                          graph: base.Graph,
                          track_labels: Optional[List[int]] = None,
                          root_labels: Optional[List[int]] = None,
                          problem_setup=None,
                          descriptors=None):
        if track_labels is None:
            # Label graph
            track_labels = base.compute_track_labels(graph)
        if root_labels is None:
            # score to connected features within same track
            score_labels = base.compute_score_labels(graph, track_labels)
            # node within each track with highest score
            root_labels = base.compute_root_labels(graph, track_labels,
                                                   score_labels)

        levels = self.conf.level_indices if self.conf.level_indices not in \
            [None, "all"] else \
            list(reversed(range(feature_manager.num_levels)))

        outputs = {}
        for level_index in levels:
            out = self.refine(keypoints_dict,
                              feature_manager.fset(level_index),
                              graph, track_labels, root_labels,
                              problem_setup=problem_setup,
                              descriptors=descriptors)
            for k, v in out.items():
                if k in outputs.keys():
                    outputs[k].append(v)
                else:
                    outputs[k] = [v]

        return outputs


class FeatureMetricKeypointAdjuster(KeypointAdjuster):
    """
    Minimize the feature-metric error along each edge within
    tentative tracks, and fix the root node to avoid drift.

    Default method in the paper.

    Additional Params:
        root_regularize_weight: Add (missing) edges towards
            the root node with this weight. If -1,
            do not add any edges.
        weight_by_sim: Whether the loss function should be scaled
            by the similarity score along each edge.
        root_edges_only: Ignore edges which are not connected to
            the root node.
        num_threads: Threads used in outer parallelism (i.e. when
            split_in_subproblems==True).
    """
    default_conf = deepcopy(KeypointAdjuster.default_conf)
    default_conf["optimizer"] = {**default_conf["optimizer"],
                                 "root_regularize_weight": -1,
                                 "weight_by_sim": True,
                                 "root_edges_only": False,
                                 "num_threads": -1}

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(self,
               keypoints_dict: base.Map_NameKeypoints,
               feature_set: features.FeatureSet,
               graph: base.Graph,
               track_labels: List[int],
               root_labels: List[int],
               problem_setup: ka.KeypointAdjustmentSetup = None,
               descriptors=None) -> dict:
        if problem_setup is None:
            problem_setup = ka.KeypointAdjustmentSetup()
            problem_setup.set_masked_nodes_constant(graph, root_labels)
        solver = ka.FeatureMetricKeypointOptimizer(
                    to_optim_ctr(self.conf.optimizer, self.callbacks),
                    problem_setup,
                    to_ctr(self.conf.interpolation))

        if self.conf.split_in_subproblems:
            # Split problem in indepent chunks
            # nodes with same labels are optimized together
            problem_labels, _ = find_problem_labels(
                                        track_labels,
                                        self.conf.max_kps_per_problem)
            solver.run(
                problem_labels,
                keypoints_dict,
                graph,
                track_labels,
                root_labels,
                feature_set)
        else:
            solver.run(
                keypoints_dict,
                graph,
                track_labels,
                root_labels,
                feature_set)
        return {"summary": solver.summary()}


class TopologicalReferenceKeypointAdjuster(KeypointAdjuster):
    """
    Optimize all keypoints within a track towards the
    node with the highest aggregated matching score using deep
    feature gradients.

    Significantly faster than FeatureMetricKeypointAdjuster
    (linear vs. quadratic), but slightly less accurate.

    Additional Params:
        num_threads: Threads used in outer parallelism (i.e. when
            split_in_subproblems==True).
    """
    default_conf = deepcopy(KeypointAdjuster.default_conf)
    default_conf["optimizer"] = {**default_conf["optimizer"],
                                 "num_threads": -1}

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(self,
               keypoints_dict: base.Map_NameKeypoints,
               feature_set: features.FeatureSet,
               graph: base.Graph,
               track_labels: List[int],
               root_labels: List[int],
               problem_setup: ka.KeypointAdjustmentSetup = None,
               descriptors=None) -> dict:
        if problem_setup is None:
            problem_setup = ka.KeypointAdjustmentSetup()
            problem_setup.set_masked_nodes_constant(graph, root_labels)
        solver = ka.TopologicalReferenceKeypointOptimizer(
                    to_optim_ctr(self.conf.optimizer, self.callbacks),
                    problem_setup,
                    to_ctr(self.conf.interpolation))

        if self.conf.split_in_subproblems:
            problem_labels, _ = find_problem_labels(
                                        track_labels,
                                        self.conf.max_kps_per_problem)
            solver.run(
                problem_labels,
                keypoints_dict,
                graph,
                track_labels,
                root_labels,
                feature_set)
        else:
            solver.run(
                keypoints_dict,
                graph,
                track_labels,
                root_labels,
                feature_set)
        return {"summary": solver.summary()}

import torch
import torch.nn.functional as F

class Keypt2SubpxKeypointAdjuster(KeypointAdjuster):
    default_conf = deepcopy(KeypointAdjuster.default_conf)
    default_conf["optimizer"] = {
        **default_conf["optimizer"],
        "num_threads": -1
    }

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(self,
           keypoints_dict: base.Map_NameKeypoints,
           feature_set: features.FeatureSet,
           graph: base.Graph,
           track_labels: List[int],
           root_labels: List[int],
           problem_setup: ka.KeypointAdjustmentSetup = None,
           descriptors: Dict[str, np.ndarray] = {}) -> dict:

        # Extract descriptors for each track and compute average descriptors
        print("Summing Descriptors...", flush=True)
        averaged_descriptors = {k:None for k in set(track_labels)}
        num_descriptors = {k:0 for k in set(track_labels)}

        for node_idx, label in enumerate(tqdm(track_labels)):
            node = graph.nodes[node_idx]
            # Accessing to FeatureMap via fmap
            img_name = graph.image_id_to_name[node.image_id]
            # Acquiring descriptor
            desc = torch.tensor(descriptors[img_name][:,node.feature_idx])
            if averaged_descriptors[label] is None:
                averaged_descriptors[label] = desc
            else:
                averaged_descriptors[label] += desc

            num_descriptors[label] += 1
        
        print("Averaging...", flush=True)
        for label in tqdm(averaged_descriptors.keys()):
            averaged_descriptors[label] /= num_descriptors[label]
        
        print("Keypoint Refinement...", flush=True)
        logsoftargmax = SpatialSoftArgmax2d(False)
        # Perform convolution with averaged descriptor for each patch
        for node_idx, track_id in enumerate(tqdm(track_labels)):
            avg_descriptor = averaged_descriptors[track_id]
            node = graph.nodes[node_idx]
            img_name = graph.image_id_to_name[node.image_id]
            fmap = feature_set.fmap(img_name)
            if fmap is not None:
                fpatch = fmap.fpatch(node.feature_idx)
                if fpatch is not None and fpatch.has_data():
                    # convert patch data to tensor
                    patch = torch.tensor(fpatch.data)
                    patch_tensor = patch.permute(2, 0, 1).unsqueeze(0)  # P x P x F -> 1 x F x P x P
                    scale = fpatch.scale

                    descriptor_tensor = avg_descriptor.view(1, -1, 1, 1)  # F -> 1 x F x 1 x 1
                    
                    P = 5
                    patch_tensor = (patch_tensor * descriptor_tensor).sum(dim=1).view(1, 1, P, P)
                    refined_coords = ((logsoftargmax(patch_tensor) - 2.) * 2.5)[0][0].cpu().numpy() # 0 ~ 4 -> -5 ~ 5, 1 x 1 x 2

                    keypoints_dict[img_name][node.feature_idx] = keypoints_dict[img_name][node.feature_idx] + refined_coords / scale

        return {"summary": "Keypt2Subpx Adjustment done!"}

def build_matching_graph(
        pairs: List[Tuple[str]],
        matches: List[np.ndarray],
        scores: Optional[List[np.ndarray]] = None) -> base.Graph:
    logger.info("Building matching graph...")
    graph = base.Graph()
    scores = scores if scores is not None else [None for _ in matches]
    for (name1, name2), m, s in zip(pairs, matches, scores):
        graph.register_matches(name1, name2, m, s)
    return graph


def extract_patchdata_from_graph(graph: base.Graph) -> Dict[str, List[int]]:
    image_to_keypoint_ids = defaultdict(list)
    for node in graph.nodes:
        image_name = graph.image_id_to_name[node.image_id]
        image_to_keypoint_ids[image_name].append(node.feature_idx)
    return dict(image_to_keypoint_ids)
