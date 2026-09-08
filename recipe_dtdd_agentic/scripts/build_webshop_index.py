"""Write the document file for WebShop's search index over the FULL catalog.

WebShop's search engine is a Lucene index built by ``pyserini`` from a
``documents.jsonl`` file, one document per product. The stock converter
(``search_engine/convert_product_file_format.py``) reads the product list
through ``web_agent_site.utils.DEFAULT_FILE_PATH``. After
``../overlay/webshop_catalog.patch`` that path points at the 120k subset,
which is right for the live environment but wrong for the index: the agent's
searches should be answered from the whole 1.18M-item release, not from the
subset it happens to be trained on.

This script therefore does not touch ``utils.py``. It overrides the
module-level path globals at runtime, loads the full files from the data
snapshot directly, and writes ``documents.jsonl`` in the shape the stock
converter produces.

Input
-----
    --snapshot-dir      directory holding the full ``items_shuffle.json`` and
                        ``items_ins_v2.json`` (the ``YWZBrandon/webshop-data``
                        Hugging Face snapshot, or an equivalent download)
    --webshop-root      the WebShop checkout; ``web_agent_site`` is imported
                        from it and ``data/items_human_ins.json`` (the full
                        12,251-goal file) is read from it
    --output-dir        where to write ``documents.jsonl``; the stock name is
                        ``search_engine/resources_full``

Output
------
``<output-dir>/documents.jsonl``: one JSON object per line with

    id          the product ASIN
    contents    lower-cased searchable text: title, description, first
                bullet point and the option names/values
    product     the full product dict

Then build the index exactly as WebShop's own ``setup.sh`` does, e.g.

    python -m pyserini.index.lucene --collection JsonCollection \\
        --input <output-dir> --index search_engine/indexes_full \\
        --generator DefaultLuceneDocumentGenerator --threads 8 \\
        --storePositions --storeDocvectors --storeRaw

(``pyserini`` needs a Java 17 runtime on ``PATH``.)

The script refuses to proceed if fewer than one million products load, which
is the symptom of the paths still resolving to a capped subset.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Below this the loaded catalog cannot be the full release.
MIN_FULL_CATALOG_SIZE = 1_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Directory with the full items_shuffle.json and items_ins_v2.json",
    )
    parser.add_argument(
        "--webshop-root",
        required=True,
        help="Path to the WebShop checkout (provides web_agent_site).",
    )
    parser.add_argument(
        "--output-dir",
        default="resources_full",
        help="Directory to write documents.jsonl into.",
    )
    return parser.parse_args()


def product_to_document(product: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one product dict into the searchable document shape.

    Args:
        product: A product entry from ``items_shuffle.json``.

    Returns:
        A dict with ``id``, ``contents`` and ``product`` keys, matching what
        WebShop's stock converter writes.
    """
    option_texts: List[str] = []
    for option_name, option_values in product.get("options", {}).items():
        option_texts.append(f"{option_name}: {', '.join(option_values)}")
    contents = " ".join(
        [
            product["Title"],
            product["Description"],
            product["BulletPoints"][0],
            ", and ".join(option_texts),
        ]
    ).lower()
    return {"id": product["asin"], "contents": contents, "product": product}


def main() -> None:
    args = parse_args()
    full_items = os.path.join(args.snapshot_dir, "items_shuffle.json")
    full_attributes = os.path.join(args.snapshot_dir, "items_ins_v2.json")
    full_human = os.path.join(
        args.webshop_root, "data", "items_human_ins.json"
    )

    # web_agent_site is imported from the checkout, not installed.
    sys.path.insert(0, args.webshop_root)
    import web_agent_site.utils as webshop_utils  # noqa: E402

    webshop_utils.DEFAULT_FILE_PATH = full_items
    webshop_utils.DEFAULT_ATTR_PATH = full_attributes
    webshop_utils.HUMAN_ATTR_PATH = full_human

    import web_agent_site.engine.engine as webshop_engine  # noqa: E402

    # The engine module may have imported the names by value at import time;
    # override its copies too.
    for name, value in (
        ("DEFAULT_FILE_PATH", full_items),
        ("DEFAULT_ATTR_PATH", full_attributes),
        ("HUMAN_ATTR_PATH", full_human),
    ):
        if hasattr(webshop_engine, name):
            setattr(webshop_engine, name, value)

    all_products, *_ = webshop_engine.load_products(filepath=full_items)
    print(f"loaded products: {len(all_products)}", flush=True)
    assert len(all_products) > MIN_FULL_CATALOG_SIZE, (
        f"only {len(all_products)} products loaded; the paths still resolve "
        "to a capped subset rather than the full release"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "documents.jsonl")
    with open(output_path, "w") as handle:
        for product in all_products:
            handle.write(json.dumps(product_to_document(product)) + "\n")
    print(f"WROTE {output_path}", flush=True)


if __name__ == "__main__":
    main()
