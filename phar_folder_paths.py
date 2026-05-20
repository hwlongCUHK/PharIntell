"""Disk-safe basename rules for post_search folder keys (no tools package import side effects)."""


def safe_folder_filename(name: str) -> str:
    """Same rule as emb_data .npy basename: spaces -> underscore, colons -> hyphen."""
    return name.replace(" ", "_").replace(":", "-")
