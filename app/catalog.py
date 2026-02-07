'''
What is this for? So we have a labs folder which has a bunch of .yml files. 
But teachers need more than filenames, they need RAM cost, topic, basic/advanced, title + description
This code builds a catalog to load the metadata of a yml file from a sidecar json file
'''
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

#describe 1 lab entry in the catalog
@dataclass
class LabMeta:
    id: str
    filename: str
    title: str
    description: str
    ram_mb: int
    topic: str
    level: str  # "basic" | "advanced"

#takes a dir path (labs dir) and returns a list of LabMeta objs
def load_catalog(labs_dir: Path) -> list[LabMeta]:
    output: list[LabMeta] = []

    for f in sorted(labs_dir.glob("*.yml")):
        lab_id = f.stem
        meta_file = labs_dir / f"{lab_id}.meta.json"

        # set defaults, so system still works if a meta is missing w/o breaking
        meta = {
            "title": lab_id,
            "description": "",
            "ram_mb": 1024,
            "topic": "misc",
            "level": "basic",
        }
        if meta_file.exists():
            meta.update(json.loads(meta_file.read_text(encoding="utf-8")))

        output.append(LabMeta(
            id=lab_id,
            filename=f.name,
            title=str(meta.get("title", lab_id)),
            description=str(meta.get("description", "")),
            ram_mb=int(meta.get("ram_mb", 1024)),
            topic=str(meta.get("topic", "misc")),
            level=str(meta.get("level", "basic")),
        ))
    return output

#this func groups labs by (topic, level) and calculates max_ram_mb
def topics_modes(catalog: list[LabMeta]):
    """
    returns dict: (topic, level) -> {max_ram_mb, labs:[ids]}
    example output:
    topic = “routing-basics”
    level = “basic”
    labs = [lab1, lab2, lab3]
    max_ram_mb = max(ram_mb among those labs)
    """
    grouped = {}
    for lab in catalog:
        key = (lab.topic, lab.level)
        if key not in grouped:
            grouped[key] = {
                "topic": lab.topic,
                "level": lab.level,
                "max_ram_mb": 0,
                "labs": []
            }
        g = grouped[key]
        g["labs"].append(lab.id)
        if lab.ram_mb > g["max_ram_mb"]:
            g["max_ram_mb"] = lab.ram_mb
    return list(grouped.values())