"""No geometry, perception, robot API, platform connection or official judging."""

import hashlib
import json
import math
from pathlib import Path

COEFFICIENTS = {"autonomous": 1.0, "teleoperation": 0.5}


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty text")


def _number(value, label, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label}: outside supported range")


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label}: expected boolean")


def _items(value, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"final_state: expected list with at most {maximum} items")
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("final_state item must be an object")
        name = item.get("id")
        _text(name, "id")
        if name in seen:
            raise ValueError(f"duplicate object id: {name}")
        seen.add(name)
    return value


def evaluate(record):
    """Calculate scores from supplied judgments, not from images or poses.

    stable_valid, shape_awarded, color_awarded and effective_inserted are
    externally reviewed labels. An unknown judgment must not be invented.
    Only a single final snapshot is used; past successes cannot accumulate.
    """
    _keys(record, {"schema_version", "run_id", "task", "control_mode",
                   "elapsed_s", "evidence_kind", "evidence_ref", "final_state"},
          "record")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    for key in ("run_id", "task", "control_mode", "evidence_kind", "evidence_ref"):
        _text(record[key], key)
    if record["control_mode"] not in COEFFICIENTS:
        raise ValueError("control_mode: unknown/mixed mode requires official clarification")
    if record["evidence_kind"] not in ("synthetic_example", "manual_review"):
        raise ValueError("evidence_kind: only synthetic_example or manual_review")
    # This is a supported-input boundary, not an official over-time penalty.
    _number(record["elapsed_s"], "elapsed_s", maximum=300)
    task = record["task"]
    if task == "brick":
        items = _items(record["final_state"], 10)
        base = shape = color = 0
        for item in items:
            _keys(item, {"id", "released", "stable_for_s", "stable_valid",
                         "shape_awarded", "color_awarded"}, "brick")
            for key in ("released", "stable_valid", "shape_awarded", "color_awarded"):
                _boolean(item[key], key)
            _number(item["stable_for_s"], "stable_for_s")
            if item["stable_for_s"] > record["elapsed_s"]:
                raise ValueError("stable_for_s cannot exceed elapsed_s")
            if item["stable_valid"]:
                if not item["released"] or item["stable_for_s"] < 3:
                    raise ValueError("stable_valid requires released and >=3s observed stability")
                base += 3
                shape += int(item["shape_awarded"])
                color += int(item["color_awarded"])
            elif item["shape_awarded"] or item["color_awarded"]:
                raise ValueError("an invalid block cannot carry awarded shape/color points")
        components = {"stable_base": base, "shape": shape, "color": color}
        # Do not infer whether color requires shape; supplied awards are reviewed.
        raw = base + shape + color
    elif task == "assembly":
        items = _items(record["final_state"], 6)
        counts = {"preset": 0, "loose": 0}
        components = {"preset": 0, "loose": 0}
        for item in items:
            _keys(item, {"id", "kind", "effective_inserted"}, "screw")
            _text(item["kind"], "kind")
            if item["kind"] not in counts:
                raise ValueError("screw kind must be preset or loose")
            _boolean(item["effective_inserted"], "effective_inserted")
            kind = item["kind"]
            counts[kind] += 1
            components[kind] += (8 if kind == "preset" else 9) * item["effective_inserted"]
        if counts["preset"] > 4 or counts["loose"] > 2:
            raise ValueError("supported provisional assembly inventory is 4 preset + 2 loose")
        raw = sum(components.values())
    else:
        raise ValueError("task must be brick or assembly")
    return {
        "run_id": record["run_id"],
        "official_judge": False,
        "basis": "supplied_final_state_annotations",
        "evidence_kind": record["evidence_kind"],
        "raw_score": raw,
        "coefficient": COEFFICIENTS[record["control_mode"]],
        "weighted_score": raw * COEFFICIENTS[record["control_mode"]],
        "components": components,
        "elapsed_s": record["elapsed_s"],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_record(path):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object,
                      parse_constant=invalid_constant)


def append_review(path, record):
    """Append one validated offline review; no claim of tamper-proof logging.

    Parent directory must already exist. Single-writer only. Repeated run_ids
    are allowed for manual review revisions; users must not count revisions as
    independent robot trials.
    """
    result = evaluate(record)
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    entry = {"schema_version": 1,
             "record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
             "record": record, "result": result}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    return result
