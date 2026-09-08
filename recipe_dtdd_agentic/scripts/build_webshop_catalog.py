"""Build the WebShop product catalog used by the paper's WebShop experiments.

WebShop ships a 1,000-product demo catalog. The experiments use a 120,000-
product catalog built from the full 1.18M-item release, and how that subset
is chosen matters more than its size:

    Human-annotated goals cover only about 1% of products. A RANDOM 120k
    slice therefore keeps about 1% of the goal-bearing products, and with them
    only a few hundred goals. Instead, keep EVERY product that has a human
    instruction, then pad with random distractors up to the target size:

        ~10,100 goal-bearing products + ~110k distractors = 120k catalog
                                                          -> ~12,250 goals

    The distractors only control search difficulty; the goals are what the
    agent is trained and evaluated on.

Input
-----
Three files from the WebShop data release (the ``YWZBrandon/webshop-data``
snapshot on the Hugging Face Hub, or an equivalent download):

    items_shuffle.json      the full product list; a JSON array of product
                            dicts, each with an ``asin`` key
    items_human_ins.json    human instructions; a JSON object mapping
                            ``asin -> list of instruction dicts``
    items_ins_v2.json       attribute annotations; a JSON object keyed by
                            ``asin`` (or a list of dicts with an ``asin`` key)

Output
------
Three files in ``--output-dir``, named with the catalog size so that several
sizes can coexist and so that WebShop's own defaults can be pointed at them
(see ``../overlay/webshop_catalog.patch``):

    items_shuffle_<N>.json      the subset catalog, shuffled
    items_human_ins_<N>.json    instructions restricted to kept products
    items_ins_v2_<N>.json       attributes restricted to kept products

The subset is deterministic for a given input and seed.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

# Seed behind the catalog used for every WebShop number in the paper.
DEFAULT_SEED = 42
DEFAULT_CATALOG_SIZE = 120_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--items", required=True, help="Path to the full items_shuffle.json."
    )
    parser.add_argument(
        "--human-instructions",
        required=True,
        help="Path to items_human_ins.json.",
    )
    parser.add_argument(
        "--attributes",
        required=True,
        help="Path to items_ins_v2.json (skipped if the file does not exist).",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the subset to."
    )
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=DEFAULT_CATALOG_SIZE,
        help="Total number of products in the subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for choosing distractors and shuffling.",
    )
    return parser.parse_args()


def split_by_goal_coverage(
    products: Sequence[Dict[str, Any]],
    goal_asins: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition products into those with a human goal and the rest.

    Args:
        products: The full product list.
        goal_asins: ASINs that have at least one human instruction.

    Returns:
        ``(goal_products, other_products)``, each preserving input order.
    """
    goal_products: List[Dict[str, Any]] = []
    other_products: List[Dict[str, Any]] = []
    for product in products:
        if product.get("asin") in goal_asins:
            goal_products.append(product)
        else:
            other_products.append(product)
    return goal_products, other_products


def restrict_attributes(
    attributes: Union[Dict[str, Any], List[Dict[str, Any]]],
    kept_asins: Set[str],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Keep only the attribute entries whose ASIN is in the subset.

    The attribute file has shipped in two shapes -- a dict keyed by ASIN and a
    list of dicts carrying an ``asin`` key -- so both are handled.

    Args:
        attributes: The parsed attribute file.
        kept_asins: ASINs present in the subset catalog.

    Returns:
        The restricted structure, in the same shape as the input.
    """
    if isinstance(attributes, dict):
        return {
            asin: value
            for asin, value in attributes.items()
            if asin in kept_asins
        }
    return [entry for entry in attributes if entry.get("asin") in kept_asins]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("loading full catalog ...", flush=True)
    with open(args.items) as handle:
        products: List[Dict[str, Any]] = json.load(handle)
    with open(args.human_instructions) as handle:
        human_instructions: Dict[str, List[Any]] = json.load(handle)
    goal_asins = set(human_instructions)
    total_goals = sum(len(v) for v in human_instructions.values())
    print(
        f"  catalog={len(products)}  goal-bearing asins={len(goal_asins)}  "
        f"goals={total_goals}"
    )

    goal_products, other_products = split_by_goal_coverage(
        products, goal_asins
    )
    print(f"  goal-bearing products found in catalog: {len(goal_products)}")

    num_distractors = max(0, args.catalog_size - len(goal_products))
    random.shuffle(other_products)
    subset = goal_products + other_products[:num_distractors]
    random.shuffle(subset)
    kept_asins = {product.get("asin") for product in subset}

    subset_instructions = {
        asin: value
        for asin, value in human_instructions.items()
        if asin in kept_asins
    }
    subset_goals = sum(len(v) for v in subset_instructions.values())
    coverage = subset_goals / max(1, len(subset)) * 100
    print(
        f"  -> subset catalog={len(subset)}  goals={subset_goals}  "
        f"({coverage:.1f}% coverage)"
    )

    size = args.catalog_size
    with open(
        os.path.join(args.output_dir, f"items_shuffle_{size}.json"), "w"
    ) as handle:
        json.dump(subset, handle)
    with open(
        os.path.join(args.output_dir, f"items_human_ins_{size}.json"), "w"
    ) as handle:
        json.dump(subset_instructions, handle)

    if os.path.exists(args.attributes):
        with open(args.attributes) as handle:
            attributes = json.load(handle)
        subset_attributes = restrict_attributes(attributes, kept_asins)
        with open(
            os.path.join(args.output_dir, f"items_ins_v2_{size}.json"), "w"
        ) as handle:
            json.dump(subset_attributes, handle)
        print(f"  attribute entries kept: {len(subset_attributes)}")

    print("SUBSET_DONE", subset_goals, flush=True)


if __name__ == "__main__":
    main()
