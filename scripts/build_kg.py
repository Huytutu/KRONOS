"""Map the 14 VinDr-CXR findings onto Radiology Gamuts Ontology (RGO) concepts.

Phase 1 of KG construction: a mapping ASSISTANT, not a blind builder. It tries
several keys in order (RadLex RID xref -> exact label -> curated synonym) and,
for any finding it cannot map, lists candidate RGO concepts for a human to pick.

Outputs:
  data/ontology/vindr_to_rgo.yaml   draft mapping (auto rows + MANUAL rows)
  + a coverage report printed to stdout.

Phase 2 (subgraph extraction over may_cause) runs only after this mapping is
verified by a human. Run: python scripts/build_kg.py
"""
import re
from collections import defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
RGO_PATH = ROOT / "data" / "RGO-2.0.owl"
SYN_PATH = ROOT / "data" / "ontology" / "synonyms.yaml"
OUT_PATH = ROOT / "data" / "ontology" / "vindr_to_rgo.yaml"
KG_PATH = ROOT / "data" / "ontology" / "causal_kg.yaml"
HOPS = 1  # neighborhood radius around the seeds (induced edges give 2-hop seed-to-seed paths)

# Tokens too generic to suggest candidates by — they match thousands of labels.
WEAK_TOKENS = {"lung", "pulmonary", "other", "lesion", "finding", "disease", "abnormality"}

# Human decisions (verified 2026-06-23) for findings RGO cannot faithfully map.
# These join the KG via RadLex anatomy, not may_cause. Editing here keeps the
# generated mapping reproducible across re-runs.
MANUAL_MAP = {
    "Consolidation": (
        "RGO has no generic consolidation node; all 5 'consolidation' nodes are "
        "specific subtypes (lobar/segmental/chronic/unilateral) sitting directly "
        "under the root. No faithful generic match -> observation-only."
    ),
    "Infiltration": (
        "RGO has no generic pulmonary infiltrate; all 'infiltrat*' nodes are "
        "non-pulmonary (leukemic/fatty/carcinomatous). -> observation-only."
    ),
    "Other lesion": (
        "VinDr catch-all grouping label, not a pathological entity. -> observation-only."
    ),
}


# --- parse RGO ---

# The RGO OWL is line-oriented and very regular, but not guaranteed well-formed
# XML (a stray malformed tag was seen), so we scan lines instead of using an XML
# parser. We only need three things per class: its id, its English label, and
# any RadLex RID cross-references.
_CLASS = re.compile(r'<owl:Class\s+rdf:about="([^"]+)"')
_LABEL = re.compile(r'<rdfs:label[^>]*>([^<]+)</rdfs:label>')
_RID_XREF = re.compile(r'<oboInOwl:hasDbXref>RADLEX:(RID\d+)</oboInOwl:hasDbXref>')
# Direct is-a parent only. The other subClassOf form wraps an owl:Restriction
# (may_cause/may_be_caused_by) and has no rdf:resource on its own line, so it
# is ignored here.
_PARENT = re.compile(r'<rdfs:subClassOf\s+rdf:resource="(rgo:\d+)"')
# may_cause / may_be_caused_by are encoded as owl:Restriction: an onProperty line
# names the relation, the next someValuesFrom line names the target concept.
_ONPROP = re.compile(r'<owl:onProperty\s+rdf:resource="(may_cause|may_be_caused_by)"')
_SOMEVAL = re.compile(r'<owl:someValuesFrom\s+rdf:resource="(rgo:\d+)"')


def parse_rgo(path):
    """Return RGO concepts as a list of dicts with:
    id, label (lowercased), rids:set, parents (is-a ids),
    causes (ids this concept may_cause), caused_by (ids that may_cause it)."""
    concepts = []
    current = None
    pending = None  # the relation of a restriction awaiting its target line
    for line in path.read_text(encoding="utf-8").splitlines():
        start = _CLASS.search(line)
        if start:
            current = {"id": start.group(1), "label": "", "rids": set(),
                       "parents": [], "causes": [], "caused_by": []}
            pending = None
            continue
        if current is None:
            continue

        label = _LABEL.search(line)
        if label and not current["label"]:
            current["label"] = label.group(1).strip().lower()

        rid = _RID_XREF.search(line)
        if rid:
            current["rids"].add(rid.group(1))

        parent = _PARENT.search(line)
        if parent:
            current["parents"].append(parent.group(1))

        prop = _ONPROP.search(line)
        if prop:
            pending = prop.group(1)
            continue

        target = _SOMEVAL.search(line)
        if target and pending:
            if pending == "may_cause":
                current["causes"].append(target.group(1))
            else:
                current["caused_by"].append(target.group(1))
            pending = None

        if "</owl:Class>" in line:
            concepts.append(current)
            current = None
            pending = None
    return concepts


# --- load findings ---

def load_findings(path):
    """The 14 findings, each as {name, rid, synonyms}. synonyms.yaml is the
    authoritative finding list (canonical name + RID + curated synonyms)."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        {"name": e["canonical"], "rid": e.get("rid"), "synonyms": e.get("synonyms", [])}
        for e in entries
    ]


# --- map ---

def map_findings(findings, concepts):
    by_rid = {}
    by_label = {}
    for c in concepts:
        for r in c["rids"]:
            by_rid.setdefault(r, c)         # first class wins on collisions
        if c["label"]:
            by_label.setdefault(c["label"], c)

    rows = []
    for f in findings:
        if f["name"] in MANUAL_MAP:
            rows.append({
                "finding": f["name"], "rid": f["rid"], "rgo_id": None,
                "status": "observation-only", "note": MANUAL_MAP[f["name"]],
            })
            continue

        match, method, via = _match(f, by_rid, by_label)
        row = {"finding": f["name"], "rid": f["rid"]}
        if match:
            row["rgo_id"] = match["id"]
            row["rgo_label"] = match["label"]
            row["method"] = method
            if via:
                row["matched_via"] = via
            row["status"] = "auto"
        else:
            row["rgo_id"] = None
            row["status"] = "MANUAL"
            row["candidates"] = candidates_for(f, concepts)
        rows.append(row)
    return rows


def _match(f, by_rid, by_label):
    """Try keys in order: RID xref -> exact label -> synonym. Returns
    (concept, method, matched_via) or (None, None, None)."""
    if f["rid"] and f["rid"] in by_rid:
        return by_rid[f["rid"]], "rid", None
    if f["name"].lower() in by_label:
        return by_label[f["name"].lower()], "label", None
    for syn in f["synonyms"]:
        if syn.lower() in by_label:
            return by_label[syn.lower()], "synonym", syn
    return None, None, None


def candidates_for(f, concepts, limit=8):
    """RGO concepts whose label shares a distinctive word with the finding or
    its synonyms — a shortlist for a human to choose from."""
    wanted = set()
    for name in [f["name"], *f["synonyms"]]:
        for word in re.findall(r"[a-z]+", name.lower()):
            if len(word) > 3 and word not in WEAK_TOKENS:
                wanted.add(word)

    scored = []
    for c in concepts:
        if not c["label"]:
            continue
        overlap = wanted & set(re.findall(r"[a-z]+", c["label"]))
        if overlap:
            scored.append((len(overlap), c["id"], c["label"]))

    scored.sort(key=lambda s: (-s[0], s[2]))
    return [{"rgo_id": cid, "label": label} for _, cid, label in scored[:limit]]


# --- phase 2: causal subgraph ---

def build_causal_subgraph(concepts, seeds, hops=HOPS):
    """From the seed RGO ids, keep every concept within `hops` of a seed on the
    may_cause graph, and the induced may_cause edges among kept concepts.
    Returns (kept_ids, edges, by_id). Each edge is (source, target) = source may_cause target."""
    by_id = {c["id"]: c for c in concepts}

    directed = defaultdict(set)   # source may_cause target
    for c in concepts:
        for t in c["causes"]:
            if t in by_id:
                directed[c["id"]].add(t)
        for s in c["caused_by"]:
            if s in by_id:
                directed[s].add(c["id"])

    undirected = defaultdict(set)
    for source, targets in directed.items():
        for t in targets:
            undirected[source].add(t)
            undirected[t].add(source)

    kept = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            nxt |= undirected[node]
        nxt -= kept
        kept |= nxt
        frontier = nxt

    edges = sorted((s, t) for s in kept for t in directed[s] if t in kept)
    # RGO's own rule: an entity that may_cause something is disorder-like; one
    # that is only an effect is observation-like. Used to prefer disorder
    # intermediates when scoring causal paths.
    roles = {cid: ("disorder" if directed[cid] else "observation") for cid in kept}
    return kept, edges, by_id, roles


def write_causal_kg(seeds, kept, edges, by_id, roles, path):
    seed_of = {rgo_id: finding for finding, rgo_id in seeds.items()}
    nodes = []
    for cid in sorted(kept):
        node = {"id": cid, "label": by_id[cid]["label"], "role": roles[cid]}
        if cid in seed_of:
            node["seed"] = seed_of[cid]
        nodes.append(node)

    doc = {
        "seeds": seeds,
        "nodes": nodes,
        "edges": [{"source": s, "target": t} for s, t in edges],
    }
    header = ("# Generated by scripts/build_kg.py (phase 2) — RGO may_cause subgraph\n"
              "# around the mapped VinDr findings. Do not hand-edit; re-run the script.\n")
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


# --- output ---

def report(rows):
    auto = [r for r in rows if r["status"] == "auto"]
    obs = [r for r in rows if r["status"] == "observation-only"]
    manual = [r for r in rows if r["status"] == "MANUAL"]

    print(f"Mapped {len(auto)}/{len(rows)} findings automatically:")
    for r in auto:
        via = f" via '{r['matched_via']}'" if "matched_via" in r else ""
        print(f"  [{r['method']}{via}] {r['finding']} -> {r['rgo_id']} ({r['rgo_label']})")

    if obs:
        print(f"\nObservation-only by decision: {len(obs)}")
        for r in obs:
            print(f"  {r['finding']} — {r['note']}")

    if manual:
        print(f"\nNeed manual decision: {len(manual)}")
        for r in manual:
            print(f"  {r['finding']} — candidates:")
            for c in r["candidates"]:
                print(f"      {c['rgo_id']}  {c['label']}")


def main():
    concepts = parse_rgo(RGO_PATH)
    print(f"Parsed {len(concepts)} RGO concepts from {RGO_PATH.name}\n")

    findings = load_findings(SYN_PATH)
    rows = map_findings(findings, concepts)

    OUT_PATH.write_text(
        yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    report(rows)
    print(f"\nDraft mapping written to {OUT_PATH}")

    # Phase 2: causal subgraph around the verified (auto-mapped) seeds.
    seeds = {r["finding"]: r["rgo_id"] for r in rows if r["status"] == "auto"}
    kept, edges, by_id, roles = build_causal_subgraph(concepts, set(seeds.values()))
    write_causal_kg(seeds, kept, edges, by_id, roles, KG_PATH)
    print(f"\nCausal subgraph ({HOPS}-hop): {len(kept)} nodes, {len(edges)} edges -> {KG_PATH}")


if __name__ == "__main__":
    main()
