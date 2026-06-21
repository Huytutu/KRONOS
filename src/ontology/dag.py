import yaml
import networkx as nx


def slugify(name):
    """Canonical name -> DAG slug (e.g. 'Nodule/Mass' -> 'nodule_mass')."""
    if not name:
        return ""
    return name.lower().replace(" ", "_").replace("/", "_")


class OntologyDAG:
    """Small curated ontology DAG for symbolic reasoning.

    Loads dag.yaml into a networkx DiGraph. Provides four reasoning roles:
    subsumption (is-a), disjointness, anatomy mapping, laterality composition.
    """

    def __init__(self, dag_path, exclusion_path=None, zones_path=None):
        with open(dag_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.graph = nx.DiGraph()

        # Add nodes with attributes
        for node in data["nodes"]:
            attrs = {"name": node["name"]}
            if "rid" in node:
                attrs["rid"] = node["rid"]
            if "type" in node:
                attrs["type"] = node["type"]
            self.graph.add_node(node["id"], **attrs)

        # Add edges with relation attribute
        for edge in data["edges"]:
            self.graph.add_edge(
                edge["source"], edge["target"], relation=edge["relation"]
            )

        # name -> id lookup (for linker bridge)
        self._name_to_id = {}
        for node_id, attrs in self.graph.nodes(data=True):
            self._name_to_id[attrs["name"]] = node_id

        # Load exclusion lists if provided
        self._exclusion_lists = None
        if exclusion_path:
            with open(exclusion_path, encoding="utf-8") as f:
                self._exclusion_lists = yaml.safe_load(f)

        # Load anatomy zones if provided
        self._zones = None
        if zones_path:
            with open(zones_path, encoding="utf-8") as f:
                self._zones = yaml.safe_load(f)

    # --- helpers ---

    def get_node(self, node_id):
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def get_node_by_name(self, name):
        return self._name_to_id.get(name)

    def resolve_slug(self, name):
        """Canonical name -> slug: known node name, else slugify fallback."""
        return self._name_to_id.get(name) or slugify(name)

    def children(self, node_id):
        """Return direct is-a children (nodes whose is-a target is node_id)."""
        kids = []
        for pred in self.graph.predecessors(node_id):
            edge = self.graph.edges[pred, node_id]
            if edge.get("relation") == "is-a":
                kids.append(pred)
        return kids

    # --- Role 1: subsumption (is-a) ---

    def reachable_is_a(self, node, target):
        """Return is-a path [node, ..., target] or None.

        Walks is-a edges from node upward. The path is the derivation
        the engine cites in its trace.
        """
        if node == target or node not in self.graph or target not in self.graph:
            return None

        path = [node]
        current = node
        while current != target:
            parent = self._is_a_parent(current)
            if parent is None:
                return None
            path.append(parent)
            if parent == target:
                return path
            current = parent
        return None

    def is_a(self, node, target):
        return self.reachable_is_a(node, target) is not None

    def _is_a_parent(self, node):
        """Return the single is-a parent of node, or None."""
        for succ in self.graph.successors(node):
            edge = self.graph.edges[node, succ]
            if edge.get("relation") == "is-a":
                return succ
        return None

    # --- Role 2: disjointness ---

    def disjoint(self, a, b):
        """True if a and b are declared disjoint-with (symmetric check)."""
        if a == b:
            return False
        # Check both directions since we store one edge
        if self.graph.has_edge(a, b):
            if self.graph.edges[a, b].get("relation") == "disjoint-with":
                return True
        if self.graph.has_edge(b, a):
            if self.graph.edges[b, a].get("relation") == "disjoint-with":
                return True
        return False

    # --- closed-world negation support ---

    def get_exclusion_list(self, finding_name):
        """Return exclusion list for a finding (by canonical name), or None."""
        if self._exclusion_lists is None:
            return None
        return self._exclusion_lists.get(finding_name)

    # --- Role 3: anatomy mapping (bbox -> anatomy node via IoU) ---

    def anatomy_of(self, bbox, img_width, img_height):
        """Return anatomy zone with highest IoU overlap, or None.

        bbox: (x1, y1, x2, y2) in pixel coords.
        Normalizes to [0,1] using img_width/img_height, then computes IoU
        against each zone in anatomy_zones.yaml.
        """
        if self._zones is None:
            return None

        # Normalize bbox to [0,1]
        nx1 = bbox[0] / img_width
        ny1 = bbox[1] / img_height
        nx2 = bbox[2] / img_width
        ny2 = bbox[3] / img_height

        best_zone = None
        best_iou = 0.0
        for zone_name, zone_data in self._zones.items():
            zb = zone_data["bbox"]
            iou = _compute_iou((nx1, ny1, nx2, ny2), zb)
            if iou > best_iou:
                best_iou = iou
                best_zone = zone_name

        if best_iou <= 0:
            return None
        return best_zone

    # --- Role 4: laterality composition ---

    def compose_laterality(self, bbox, img_width, img_height):
        """Return laterality based on bbox position relative to image midline.

        PA CXR convention: patient's left = image right.
        Returns "left", "right", "bilateral", or "midline".
        """
        nx1 = bbox[0] / img_width
        nx2 = bbox[2] / img_width
        center_x = (nx1 + nx2) / 2
        width = nx2 - nx1
        midline = 0.5

        # Wide bbox spanning both sides
        if nx1 < midline - 0.05 and nx2 > midline + 0.05 and width > 0.4:
            return "bilateral"

        # Narrow bbox around center
        if abs(center_x - midline) < 0.12 and width < 0.35:
            return "midline"

        # PA convention: image left = patient right, image right = patient left
        if center_x < midline:
            return "right"
        return "left"


def _compute_iou(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union
