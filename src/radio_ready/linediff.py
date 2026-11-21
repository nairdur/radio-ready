"""Word-level line diffs, so rewrites can be shared without the original lyrics.

diff(original, new) -> a list of [start, end, replacement]: tokens original[start:end]
(whitespace tokens) are replaced by `replacement` (words joined by spaces, "" to delete).
Only the changed words are stored; the original line is needed to apply it:

    diff("Bought matching diamonds for six of my bitches",
         "Bought matching diamonds for six of my sistas")  ->  [[7, 8, "sistas"]]

When a line cannot be rebuilt exactly from token edits (unusual spacing), the diff is
{"text": new} instead. apply(original, diff) returns the new line; the round trip
apply(original, diff(original, new)) == new is checked when a diff is made.
"""

import difflib


def diff(original: str, new: str) -> list | dict:
    a, b = original.split(" "), new.split(" ")
    ops = [[i1, i2, " ".join(b[j1:j2])]
           for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False)
           .get_opcodes() if op != "equal"]
    return ops if apply(original, ops) == new else {"text": new}


def apply(original: str, ops: list | dict) -> str:
    if isinstance(ops, dict):
        return ops["text"]
    tokens = original.split(" ")
    for start, end, replacement in sorted(ops, key=lambda o: -o[0]):
        tokens[start:end] = replacement.split(" ") if replacement else []
    return " ".join(tokens)
