"""stage_model.py — one-time: download the spaCy model and write it into the Volume.

Run this ONCE from an environment that can reach the internet (or the Artifactory
mirror). Production never downloads: the redzone firewalls spacy.io, so a runtime
`spacy.load("en_core_web_lg")` hangs instead of failing fast. Stage 01b loads the
model by PATH from the Volume this writes.

    python stage_model.py                      # uses config.yml; SKIPS if already staged
    python stage_model.py /Volumes/.../custom  # explicit destination
    python stage_model.py --force              # re-download and overwrite the Volume copy
"""

import os
import sys


def _already_staged(dest):
    """True if a usable spaCy model already sits at dest — so we never download twice.

    A staged model is a directory containing a meta.json; probe that cheaply, then
    confirm it actually loads before trusting it.
    """
    import spacy

    if not os.path.isfile(os.path.join(dest, "meta.json")):
        return False
    try:
        spacy.load(dest)
        return True
    except Exception as e:
        print(f"[stage] {dest} exists but does not load ({e}) — will re-stage")
        return False


def stage(model_name, dest, force=False):
    import spacy

    if not force and _already_staged(dest):
        print(f"[stage] SKIP download — {model_name} already staged at {dest}")
        print("[stage] OK — stage 01b can load it offline. (pass force=True to re-stage)")
        return dest

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
    argv = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    dest = argv[0] if argv else pc["model_path"]
    stage(pc.get("spacy_model", "en_core_web_lg"), dest, force=force)


if __name__ == "__main__":
    main()
