"""stage_model.py — one-time: download the spaCy model and write it into the Volume.

Run this ONCE from an environment that can reach the internet (or the Artifactory
mirror). Production never downloads: the redzone firewalls spacy.io, so a runtime
`spacy.load("en_core_web_lg")` hangs instead of failing fast. Stage 01b loads the
model by PATH from the Volume this writes.

    python stage_model.py                      # uses config.yml
    python stage_model.py /Volumes/.../custom  # explicit destination
"""

import os
import sys


def stage(model_name, dest):
    import spacy

    print(f"[stage] loading {model_name} ...")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"[stage] {model_name} not installed — downloading")
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)

    # to_disk does NOT create parent directories; it raises FileNotFoundError.
    os.makedirs(dest, exist_ok=True)
    nlp.to_disk(dest)
    print(f"[stage] wrote {model_name} -> {dest}")

    print("[stage] verifying the model loads back from the Volume path ...")
    spacy.load(dest)
    print("[stage] OK — stage 01b can now load it offline.")
    return dest


def main():
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "config.yml")))
    pc = cfg["pii_redaction"]
    dest = sys.argv[1] if len(sys.argv) > 1 else pc["model_path"]
    stage(pc.get("spacy_model", "en_core_web_lg"), dest)


if __name__ == "__main__":
    main()
