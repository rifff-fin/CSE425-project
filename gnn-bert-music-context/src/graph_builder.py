from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data


class MusicGraphBuilder:
    """Create a PyTorch Geometric graph from audio segment descriptors."""

    def __init__(self, similarity_threshold: float = 0.7, temporal_window: int = 3) -> None:
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window

    @staticmethod
    def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"Expected 1D vectors, got {x.shape} and {y.shape}")
        if x.size == 0 or y.size == 0:
            raise ValueError("Cannot compute cosine similarity for empty vectors.")

        x_norm = x / (np.linalg.norm(x) + 1e-8)
        y_norm = y / (np.linalg.norm(y) + 1e-8)
        return float(np.dot(x_norm, y_norm))

    @staticmethod
    def segment_to_node_vector(chroma: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
        """Construct a node feature vector from chroma and MFCC statistics."""
        if chroma.ndim != 2 or mfcc.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got chroma={chroma.shape}, mfcc={mfcc.shape}")

        chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
        feature = np.concatenate([chroma_mean, mfcc_mean], axis=0)
        if feature.size == 0:
            raise ValueError("Segment node feature vector is empty.")
        return feature.astype(np.float32)

    def build_graph(self, segment_vectors: Sequence[np.ndarray]) -> Data:
        """Build a graph with temporal adjacency and cosine thresholded edges."""
        if len(segment_vectors) == 0:
            raise ValueError("No segment vectors provided to build a graph.")

        node_features = np.stack(segment_vectors, axis=0).astype(np.float32)
        num_nodes = node_features.shape[0]
        edge_index: List[List[int]] = []
        edge_attr: List[float] = []
        seen_edges: set[tuple[int, int]] = set()

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                temporal_gap = abs(i - j)
                sim = self.cosine_similarity(node_features[i], node_features[j])
                should_link = temporal_gap <= self.temporal_window or sim > self.similarity_threshold

                if not should_link:
                    continue

                for src, dst in [(i, j), (j, i)]:
                    edge_key = (int(src), int(dst))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    edge_index.extend([[src, dst]])
                    edge_attr.append(float(sim))

        if len(edge_index) == 0:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)

        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
        )
        graph.num_nodes = num_nodes
        return graph

    def build_from_segments(self, segment_features: Iterable[np.ndarray]) -> Data:
        """Convenience wrapper for graph creation from segment-level descriptors."""
        vectors = list(segment_features)
        return self.build_graph(vectors)


if __name__ == "__main__":
    builder = MusicGraphBuilder(similarity_threshold=0.7, temporal_window=3)

    segment_vectors = [
        np.random.randn(32).astype(np.float32),
        np.random.randn(32).astype(np.float32),
        np.random.randn(32).astype(np.float32),
        np.random.randn(32).astype(np.float32),
    ]

    graph = builder.build_graph(segment_vectors)
    print(f"Node tensor shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge attribute shape: {graph.edge_attr.shape}")
