"""
Unified graph theory metrics using NetworkX.

Consolidates metrics from both pipelines:
- mrtrix3_demon_addon/scripts/generate_standardized_report.py (PureNumpyGraphMetrics)
- nilearn_RSB_analysis_pipeline/scripts/06_graph_metrics.py (GraphAnalyzer)

All metrics use NetworkX for consistency across both DTI and fMRI pipelines.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import networkx as nx
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class GraphMetricsResult:
    """Container for graph metrics results."""

    global_metrics: Dict[str, Union[int, float, None]] = field(default_factory=dict)
    nodal_metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.metadata,
            'global_metrics': self.global_metrics,
            'nodal_metrics': {k: v.tolist() for k, v in self.nodal_metrics.items()},
            'warnings': self.warnings,
        }

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved graph metrics to {filepath}")


class GraphMetrics:
    """
    Compute graph theory metrics from connectivity matrices.

    Uses NetworkX for all computations to ensure consistency between
    DTI and fMRI pipelines.

    Example:
        >>> gm = GraphMetrics()
        >>> results = gm.compute_all(connectivity_matrix, roi_labels)
        >>> print(results.global_metrics['global_efficiency'])
        >>> results.to_json("metrics.json")
    """

    def __init__(
        self,
        n_random_networks: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize GraphMetrics.

        Args:
            n_random_networks: Number of random networks for small-world calculation
            seed: Random seed for reproducibility
        """
        self.n_random_networks = n_random_networks
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def compute_all(
        self,
        matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        modality: Optional[str] = None,
        atlas: Optional[str] = None,
        subject_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GraphMetricsResult:
        """
        Compute all graph metrics from a connectivity matrix.

        Args:
            matrix: Symmetric connectivity matrix (n_rois x n_rois)
            labels: Optional ROI labels
            modality: 'dwi' or 'fmri' for metadata
            atlas: Atlas name for metadata
            subject_id: Subject ID for metadata
            session_id: Session ID for metadata

        Returns:
            GraphMetricsResult with global and nodal metrics
        """
        result = GraphMetricsResult()
        result.metadata = {
            'processing_date': datetime.now().isoformat(),
            'n_random_networks': self.n_random_networks,
        }
        if modality:
            result.metadata['modality'] = modality
        if atlas:
            result.metadata['atlas'] = atlas
        if subject_id:
            result.metadata['subject_id'] = subject_id
        if session_id:
            result.metadata['session_id'] = session_id

        # Validate matrix
        if not self._validate_matrix(matrix, result):
            return result

        # Ensure symmetric
        matrix = np.maximum(matrix, matrix.T)

        # Create NetworkX graph
        G = self._create_graph(matrix)

        # Compute global metrics
        result.global_metrics.update(self._basic_metrics(matrix, G))
        result.global_metrics.update(self._clustering_metrics(G, result))
        result.global_metrics.update(self._efficiency_metrics(G, result))
        result.global_metrics.update(self._small_world_metrics(G, result))
        result.global_metrics.update(self._topology_metrics(G, result))

        # Compute nodal metrics
        result.nodal_metrics = self._nodal_metrics(G, labels)

        return result

    def _validate_matrix(
        self,
        matrix: np.ndarray,
        result: GraphMetricsResult,
    ) -> bool:
        """Validate connectivity matrix."""
        if matrix is None:
            result.warnings.append("Matrix is None")
            return False

        if matrix.ndim != 2:
            result.warnings.append(f"Matrix is not 2D: {matrix.ndim}D")
            return False

        if matrix.shape[0] != matrix.shape[1]:
            result.warnings.append(f"Matrix is not square: {matrix.shape}")
            return False

        if np.any(np.isnan(matrix)):
            n_nan = np.sum(np.isnan(matrix))
            result.warnings.append(f"Matrix contains {n_nan} NaN values - replacing with 0")
            matrix[np.isnan(matrix)] = 0

        if np.any(np.isinf(matrix)):
            n_inf = np.sum(np.isinf(matrix))
            result.warnings.append(f"Matrix contains {n_inf} Inf values - replacing with 0")
            matrix[np.isinf(matrix)] = 0

        return True

    def _create_graph(self, matrix: np.ndarray) -> nx.Graph:
        """Create NetworkX graph from connectivity matrix."""
        G = nx.Graph()
        n_nodes = matrix.shape[0]
        G.add_nodes_from(range(n_nodes))

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                weight = matrix[i, j]
                if weight != 0:
                    # Use absolute value for weights (both pipelines do this)
                    G.add_edge(i, j, weight=abs(weight))

        return G

    def _basic_metrics(
        self,
        matrix: np.ndarray,
        G: nx.Graph,
    ) -> dict:
        """Compute basic connectivity metrics."""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_possible = n_nodes * (n_nodes - 1) / 2

        metrics = {
            'n_nodes': int(n_nodes),
            'total_connections': int(n_edges),
            'connection_density': round(n_edges / n_possible, 6) if n_possible > 0 else 0.0,
            'sparsity': round(1 - (n_edges / n_possible), 6) if n_possible > 0 else 1.0,
        }

        # Edge weight statistics
        if n_edges > 0:
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            metrics['total_weight'] = round(sum(weights), 3)
            metrics['mean_edge_weight'] = round(float(np.mean(weights)), 3)
            metrics['std_edge_weight'] = round(float(np.std(weights)), 3)
        else:
            metrics['total_weight'] = 0.0
            metrics['mean_edge_weight'] = 0.0
            metrics['std_edge_weight'] = 0.0

        # Node strength statistics (weighted degree)
        strengths = [
            sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            for node in G.nodes()
        ]
        if strengths:
            metrics['mean_node_strength'] = round(float(np.mean(strengths)), 3)
            metrics['std_node_strength'] = round(float(np.std(strengths)), 3)
            metrics['max_node_strength'] = round(float(np.max(strengths)), 3)
        else:
            metrics['mean_node_strength'] = 0.0
            metrics['std_node_strength'] = 0.0
            metrics['max_node_strength'] = 0.0

        # Degree statistics
        degrees = [G.degree(node) for node in G.nodes()]
        if degrees:
            metrics['mean_degree'] = round(float(np.mean(degrees)), 3)
            metrics['std_degree'] = round(float(np.std(degrees)), 3)
            metrics['max_degree'] = int(np.max(degrees))
        else:
            metrics['mean_degree'] = 0.0
            metrics['std_degree'] = 0.0
            metrics['max_degree'] = 0

        # Strength-degree correlation
        if len(strengths) > 1 and len(degrees) > 1:
            try:
                corr, _ = stats.pearsonr(strengths, degrees)
                metrics['strength_degree_correlation'] = round(float(corr), 6) if not np.isnan(corr) else 0.0
            except Exception:
                metrics['strength_degree_correlation'] = 0.0
        else:
            metrics['strength_degree_correlation'] = 0.0

        return metrics

    def _clustering_metrics(
        self,
        G: nx.Graph,
        result: GraphMetricsResult,
    ) -> dict:
        """Compute clustering coefficients."""
        metrics = {}

        try:
            # Binary clustering (unweighted)
            binary_G = nx.Graph()
            binary_G.add_nodes_from(G.nodes())
            binary_G.add_edges_from(G.edges())
            binary_clustering = nx.average_clustering(binary_G)
            metrics['binary_clustering_coefficient'] = round(float(binary_clustering), 6)

            # Weighted clustering
            weighted_clustering = nx.average_clustering(G, weight='weight')
            metrics['weighted_clustering_coefficient'] = round(float(weighted_clustering), 6)

            # For backward compatibility
            metrics['mean_clustering_coefficient'] = round(float(binary_clustering), 6)

        except Exception as e:
            result.warnings.append(f"Could not compute clustering: {e}")
            metrics['binary_clustering_coefficient'] = 0.0
            metrics['weighted_clustering_coefficient'] = 0.0
            metrics['mean_clustering_coefficient'] = 0.0

        return metrics

    def _efficiency_metrics(
        self,
        G: nx.Graph,
        result: GraphMetricsResult,
    ) -> dict:
        """Compute path length and efficiency metrics."""
        metrics = {}

        try:
            if G.number_of_edges() == 0:
                metrics['characteristic_path_length'] = None
                metrics['global_efficiency'] = 0.0
                metrics['local_efficiency'] = 0.0
                return metrics

            if nx.is_connected(G):
                work_graph = G
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                work_graph = G.subgraph(largest_cc).copy()
                result.warnings.append(
                    f"Graph is disconnected. Using largest component "
                    f"({len(largest_cc)}/{G.number_of_nodes()} nodes)"
                )

            if work_graph.number_of_nodes() > 1:
                # Characteristic path length
                path_length = nx.average_shortest_path_length(work_graph)
                metrics['characteristic_path_length'] = round(float(path_length), 6)

                # Global efficiency
                global_eff = nx.global_efficiency(work_graph)
                metrics['global_efficiency'] = round(float(global_eff), 6)

                # Local efficiency
                local_eff = self._compute_local_efficiency(work_graph)
                metrics['local_efficiency'] = round(float(local_eff), 6)
            else:
                metrics['characteristic_path_length'] = None
                metrics['global_efficiency'] = 0.0
                metrics['local_efficiency'] = 0.0

        except Exception as e:
            result.warnings.append(f"Could not compute efficiency metrics: {e}")
            metrics['characteristic_path_length'] = None
            metrics['global_efficiency'] = 0.0
            metrics['local_efficiency'] = 0.0

        return metrics

    def _compute_local_efficiency(self, G: nx.Graph) -> float:
        """Compute local efficiency of the network."""
        local_effs = []

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                subgraph = G.subgraph(neighbors)
                if subgraph.number_of_edges() > 0:
                    local_eff = nx.global_efficiency(subgraph)
                else:
                    local_eff = 0.0
            else:
                local_eff = 0.0
            local_effs.append(local_eff)

        return float(np.mean(local_effs)) if local_effs else 0.0

    def _small_world_metrics(
        self,
        G: nx.Graph,
        result: GraphMetricsResult,
    ) -> dict:
        """Compute small-world metrics."""
        metrics = {}

        try:
            if G.number_of_edges() == 0:
                metrics['small_worldness'] = 0.0
                metrics['normalized_clustering'] = 0.0
                metrics['normalized_path_length'] = 0.0
                return metrics

            # Get connected component
            if nx.is_connected(G):
                work_graph = G
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                work_graph = G.subgraph(largest_cc).copy()

            if work_graph.number_of_nodes() < 4:
                metrics['small_worldness'] = 0.0
                metrics['normalized_clustering'] = 0.0
                metrics['normalized_path_length'] = 0.0
                return metrics

            # Real network metrics
            real_clustering = nx.average_clustering(work_graph)
            real_path_length = nx.average_shortest_path_length(work_graph)

            # Generate random networks (Erdos-Renyi)
            n_nodes = work_graph.number_of_nodes()
            n_edges = work_graph.number_of_edges()
            p = 2 * n_edges / (n_nodes * (n_nodes - 1))

            random_clusterings = []
            random_path_lengths = []

            for _ in range(self.n_random_networks):
                random_G = nx.erdos_renyi_graph(n_nodes, p)
                if random_G.number_of_edges() > 0 and nx.is_connected(random_G):
                    random_clusterings.append(nx.average_clustering(random_G))
                    random_path_lengths.append(nx.average_shortest_path_length(random_G))

            if random_clusterings and random_path_lengths:
                mean_random_clustering = np.mean(random_clusterings)
                mean_random_path_length = np.mean(random_path_lengths)

                # Normalized metrics
                gamma = real_clustering / mean_random_clustering if mean_random_clustering > 0 else 0
                lambda_ = real_path_length / mean_random_path_length if mean_random_path_length > 0 else 0

                # Small-worldness (sigma = gamma / lambda)
                sigma = gamma / lambda_ if lambda_ > 0 else 0

                metrics['small_worldness'] = round(float(sigma), 6)
                metrics['normalized_clustering'] = round(float(gamma), 6)
                metrics['normalized_path_length'] = round(float(lambda_), 6)
            else:
                result.warnings.append("Could not generate valid random networks for small-world calculation")
                metrics['small_worldness'] = 0.0
                metrics['normalized_clustering'] = 0.0
                metrics['normalized_path_length'] = 0.0

        except Exception as e:
            result.warnings.append(f"Could not compute small-world metrics: {e}")
            metrics['small_worldness'] = 0.0
            metrics['normalized_clustering'] = 0.0
            metrics['normalized_path_length'] = 0.0

        return metrics

    def _topology_metrics(
        self,
        G: nx.Graph,
        result: GraphMetricsResult,
    ) -> dict:
        """Compute network topology metrics."""
        metrics = {}

        # Assortativity
        try:
            if G.number_of_edges() > 0:
                assortativity = nx.degree_assortativity_coefficient(G)
                metrics['assortativity'] = round(float(assortativity), 6) if not np.isnan(assortativity) else 0.0
            else:
                metrics['assortativity'] = 0.0
        except Exception as e:
            result.warnings.append(f"Could not compute assortativity: {e}")
            metrics['assortativity'] = 0.0

        # Modularity (using greedy modularity communities)
        try:
            if G.number_of_edges() > 0:
                communities = nx.community.greedy_modularity_communities(G, weight='weight')
                modularity = nx.community.modularity(G, communities, weight='weight')
                metrics['modularity'] = round(float(modularity), 6)
                metrics['n_communities'] = len(communities)
            else:
                metrics['modularity'] = 0.0
                metrics['n_communities'] = 0
        except Exception as e:
            result.warnings.append(f"Could not compute modularity: {e}")
            metrics['modularity'] = 0.0
            metrics['n_communities'] = 0

        # Rich-club coefficient (at mean degree)
        try:
            if G.number_of_edges() > 0:
                degrees = [G.degree(n) for n in G.nodes()]
                mean_k = int(np.mean(degrees))
                if mean_k > 0:
                    rc_dict = nx.rich_club_coefficient(G, normalized=False)
                    if mean_k in rc_dict:
                        metrics['rich_club_coefficient'] = round(float(rc_dict[mean_k]), 6)
                    else:
                        # Use closest available k
                        available_k = sorted(rc_dict.keys())
                        if available_k:
                            closest_k = min(available_k, key=lambda x: abs(x - mean_k))
                            metrics['rich_club_coefficient'] = round(float(rc_dict[closest_k]), 6)
                        else:
                            metrics['rich_club_coefficient'] = 0.0
                else:
                    metrics['rich_club_coefficient'] = 0.0
            else:
                metrics['rich_club_coefficient'] = 0.0
        except Exception as e:
            result.warnings.append(f"Could not compute rich-club coefficient: {e}")
            metrics['rich_club_coefficient'] = 0.0

        return metrics

    def _nodal_metrics(
        self,
        G: nx.Graph,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute per-node metrics."""
        n_nodes = G.number_of_nodes()

        if labels is None:
            labels = [f"ROI_{i:03d}" for i in range(n_nodes)]

        nodal = {
            'labels': np.array(labels),
            'degree': np.zeros(n_nodes),
            'strength': np.zeros(n_nodes),
            'clustering': np.zeros(n_nodes),
            'betweenness_centrality': np.zeros(n_nodes),
            'local_efficiency': np.zeros(n_nodes),
        }

        # Degree and strength
        for i, node in enumerate(G.nodes()):
            nodal['degree'][i] = G.degree(node)
            nodal['strength'][i] = sum(
                G[node][neighbor]['weight'] for neighbor in G.neighbors(node)
            )

        # Clustering coefficient (per node)
        clustering_dict = nx.clustering(G, weight='weight')
        for i, node in enumerate(G.nodes()):
            nodal['clustering'][i] = clustering_dict[node]

        # Betweenness centrality
        try:
            bc_dict = nx.betweenness_centrality(G, weight='weight')
            for i, node in enumerate(G.nodes()):
                nodal['betweenness_centrality'][i] = bc_dict[node]
        except Exception:
            pass  # Leave as zeros

        # Local efficiency (per node)
        for i, node in enumerate(G.nodes()):
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                subgraph = G.subgraph(neighbors)
                if subgraph.number_of_edges() > 0:
                    nodal['local_efficiency'][i] = nx.global_efficiency(subgraph)

        return nodal

    def nodal_metrics_to_dataframe(
        self,
        nodal_metrics: Dict[str, np.ndarray],
    ) -> 'pd.DataFrame':
        """Convert nodal metrics to pandas DataFrame."""
        import pandas as pd

        df_dict = {}
        for key, values in nodal_metrics.items():
            if isinstance(values, np.ndarray):
                df_dict[key] = values
            else:
                df_dict[key] = [values] * len(nodal_metrics.get('labels', []))

        return pd.DataFrame(df_dict)
