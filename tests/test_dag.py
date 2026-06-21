"""Tests for OntologyDAG: structural integrity + four reasoning roles."""

import pytest
from src.ontology.dag import OntologyDAG

DAG_PATH = "data/ontology/dag.yaml"
EXCLUSION_PATH = "data/ontology/exclusion_lists.yaml"
ZONES_PATH = "data/ontology/anatomy_zones.yaml"

VINDR_FINDINGS = [
    "aortic_enlargement", "atelectasis", "calcification", "cardiomegaly",
    "consolidation", "ild", "infiltration", "lung_opacity", "nodule_mass",
    "other_lesion", "pleural_effusion", "pleural_thickening", "pneumothorax",
    "pulmonary_fibrosis",
]

VINDR_CANONICAL_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]


@pytest.fixture(scope="module")
def dag():
    return OntologyDAG(DAG_PATH, EXCLUSION_PATH, ZONES_PATH)


@pytest.fixture(scope="module")
def dag_no_exclusion():
    return OntologyDAG(DAG_PATH)


# ============================================================
# Task 2: loader
# ============================================================

class TestLoader:
    def test_loads_without_error(self, dag):
        assert dag is not None

    def test_node_count(self, dag):
        assert len(dag.graph.nodes) == 28

    @pytest.mark.parametrize("node_id", VINDR_FINDINGS)
    def test_every_finding_exists(self, dag, node_id):
        node = dag.get_node(node_id)
        assert node is not None

    def test_get_node_returns_dict(self, dag):
        node = dag.get_node("cardiomegaly")
        assert isinstance(node, dict)
        assert "name" in node

    def test_get_node_unknown_returns_none(self, dag):
        assert dag.get_node("xyzzy") is None


# ============================================================
# Task 3: structural integrity
# ============================================================

class TestStructure:
    def test_is_a_subgraph_is_acyclic(self, dag):
        import networkx as nx
        isa_edges = [(u, v) for u, v, d in dag.graph.edges(data=True)
                     if d.get("relation") == "is-a"]
        isa_graph = nx.DiGraph(isa_edges)
        assert nx.is_directed_acyclic_graph(isa_graph)

    def test_single_root(self, dag):
        assert dag.get_node("abnormality") is not None

    @pytest.mark.parametrize("finding", VINDR_FINDINGS)
    def test_every_finding_reaches_root(self, dag, finding):
        path = dag.reachable_is_a(finding, "abnormality")
        assert path is not None, f"{finding} cannot reach root"

    @pytest.mark.parametrize("finding", VINDR_FINDINGS)
    def test_findings_are_leaves(self, dag, finding):
        kids = dag.children(finding)
        assert len(kids) == 0, f"{finding} has is-a children: {kids}"


# ============================================================
# Task 4: Role 1 — subsumption (is-a)
# ============================================================

class TestSubsumption:
    def test_direct_is_a(self, dag):
        path = dag.reachable_is_a("cardiomegaly", "cardiac_abnormality")
        assert path == ["cardiomegaly", "cardiac_abnormality"]

    def test_transitive_is_a(self, dag):
        path = dag.reachable_is_a("cardiomegaly", "abnormality")
        assert path == ["cardiomegaly", "cardiac_abnormality", "abnormality"]

    def test_not_is_a(self, dag):
        path = dag.reachable_is_a("cardiomegaly", "pulmonary_abnormality")
        assert path is None

    def test_is_a_bool(self, dag):
        assert dag.is_a("cardiomegaly", "cardiac_abnormality") is True
        assert dag.is_a("cardiomegaly", "pulmonary_abnormality") is False

    def test_self_is_not_ancestor(self, dag):
        assert dag.reachable_is_a("cardiomegaly", "cardiomegaly") is None

    def test_pulmonary_branch(self, dag):
        path = dag.reachable_is_a("consolidation", "pulmonary_abnormality")
        assert path == ["consolidation", "airspace_abnormality", "pulmonary_abnormality"]

    def test_airspace_subgroup(self, dag):
        path = dag.reachable_is_a("consolidation", "airspace_abnormality")
        assert path == ["consolidation", "airspace_abnormality"]

    def test_interstitial_subgroup(self, dag):
        path = dag.reachable_is_a("ild", "interstitial_abnormality")
        assert path == ["ild", "interstitial_abnormality"]

    def test_pleural_branch(self, dag):
        path = dag.reachable_is_a("pneumothorax", "pleural_abnormality")
        assert path == ["pneumothorax", "pleural_abnormality"]

    def test_vascular_branch(self, dag):
        path = dag.reachable_is_a("aortic_enlargement", "vascular_abnormality")
        assert path == ["aortic_enlargement", "vascular_abnormality"]


# ============================================================
# Task 5: Role 2 — disjointness
# ============================================================

class TestDisjointness:
    def test_declared_pair(self, dag):
        assert dag.disjoint("pneumothorax", "pleural_effusion") is True

    def test_symmetric(self, dag):
        assert dag.disjoint("pleural_effusion", "pneumothorax") is True

    def test_self_not_disjoint(self, dag):
        assert dag.disjoint("pneumothorax", "pneumothorax") is False

    def test_undeclared_pair(self, dag):
        assert dag.disjoint("cardiomegaly", "atelectasis") is False


# ============================================================
# Task 7: exclusion lists
# ============================================================

class TestExclusionLists:
    def test_known_finding_returns_list(self, dag):
        excl = dag.get_exclusion_list("Cardiomegaly")
        assert isinstance(excl, list)
        assert len(excl) > 0

    def test_unknown_finding_returns_none(self, dag):
        assert dag.get_exclusion_list("xyzzy") is None

    def test_no_exclusion_file_returns_none(self, dag_no_exclusion):
        assert dag_no_exclusion.get_exclusion_list("Cardiomegaly") is None

    def test_airspace_group_cross_exclusion(self, dag):
        # Consolidation exclusion list includes infiltration and lung_opacity
        excl = dag.get_exclusion_list("Consolidation")
        assert "consolidation" in excl
        assert "infiltration" in excl
        assert "lung_opacity" in excl

    def test_interstitial_group_cross_exclusion(self, dag):
        excl = dag.get_exclusion_list("ILD")
        assert "ild" in excl
        assert "pulmonary_fibrosis" in excl

    def test_every_finding_has_exclusion_list(self, dag):
        for name in VINDR_CANONICAL_NAMES:
            excl = dag.get_exclusion_list(name)
            assert excl is not None, f"Missing exclusion list for {name}"


# ============================================================
# located-in edges
# ============================================================

class TestLocatedIn:
    def test_cardiomegaly_in_heart(self, dag):
        assert dag.graph.has_edge("cardiomegaly", "heart")
        assert dag.graph.edges["cardiomegaly", "heart"]["relation"] == "located-in"

    def test_consolidation_in_lung(self, dag):
        assert dag.graph.has_edge("consolidation", "lung")

    def test_pleural_effusion_in_pleural_space(self, dag):
        assert dag.graph.has_edge("pleural_effusion", "pleural_space")


# ============================================================
# Task 7b: linker <-> DAG bridge
# ============================================================

class TestLinkerBridge:
    @pytest.mark.parametrize("name", VINDR_CANONICAL_NAMES)
    def test_canonical_name_resolves(self, dag, name):
        node_id = dag.get_node_by_name(name)
        assert node_id is not None, f"canonical '{name}' not found in DAG"

    def test_unknown_name_returns_none(self, dag):
        assert dag.get_node_by_name("xyzzy") is None


# ============================================================
# Task 8: Role 3 — anatomy_of(bbox) via IoU
# ============================================================

class TestAnatomyMapping:
    def test_right_lung_bbox(self, dag):
        # bbox clearly in right lung area (normalized)
        result = dag.anatomy_of((0.15, 0.30, 0.35, 0.60), 1.0, 1.0)
        assert result == "right_lung"

    def test_left_lung_bbox(self, dag):
        result = dag.anatomy_of((0.65, 0.30, 0.85, 0.60), 1.0, 1.0)
        assert result == "left_lung"

    def test_heart_bbox(self, dag):
        # central, lower half
        result = dag.anatomy_of((0.40, 0.50, 0.60, 0.70), 1.0, 1.0)
        assert result == "heart"

    def test_no_overlap_returns_none(self, dag):
        # bbox completely outside all zones
        result = dag.anatomy_of((0.0, 0.95, 0.05, 1.0), 1.0, 1.0)
        assert result is None

    def test_pixel_coords_normalized(self, dag):
        # bbox in pixel coords for a 2080x2336 image, in right lung area
        result = dag.anatomy_of((200, 500, 700, 1400), 2080, 2336)
        assert result == "right_lung"

    def test_no_zones_returns_none(self, dag_no_exclusion):
        result = dag_no_exclusion.anatomy_of((0.15, 0.30, 0.35, 0.60), 1.0, 1.0)
        assert result is None


# ============================================================
# Task 9: Role 4 — compose_laterality
# ============================================================

class TestLaterality:
    def test_left_side(self, dag):
        # bbox on patient's left = image right (PA convention)
        result = dag.compose_laterality((0.65, 0.30, 0.85, 0.60), 1.0, 1.0)
        assert result == "left"

    def test_right_side(self, dag):
        # bbox on patient's right = image left
        result = dag.compose_laterality((0.15, 0.30, 0.35, 0.60), 1.0, 1.0)
        assert result == "right"

    def test_bilateral(self, dag):
        # bbox spanning both sides
        result = dag.compose_laterality((0.10, 0.30, 0.90, 0.60), 1.0, 1.0)
        assert result == "bilateral"

    def test_midline(self, dag):
        # bbox centered narrowly around midline
        result = dag.compose_laterality((0.40, 0.30, 0.60, 0.70), 1.0, 1.0)
        assert result == "midline"

    def test_pixel_coords(self, dag):
        # right lung in pixel coords (patient's right = image left)
        result = dag.compose_laterality((200, 500, 700, 1400), 2080, 2336)
        assert result == "right"
