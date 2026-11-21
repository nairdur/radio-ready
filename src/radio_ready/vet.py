"""Quick human vetting of synthetic rewrites, one line at a time (tkinter, no extra installs).

Reads and writes annotations/pair_edits.csv (radio_ready.pairs; lines are stored as diffs and
filled in from the section records). Each screen shows one row:
the original line with its explicit words highlighted, and the proposed rewrite in an edit
field. Change the text if you want (it is saved as your edit), then give a verdict:

  Enter          like
  Ctrl+Enter     superlike
  Esc            dislike
  Up / Down      back to the previous line / skip this line (no verdict)
  Tab            context: the 4-line stanza, original and rewritten (closes on every line)
  Alt+1 ... 8    replace the end word with rhyme suggestion 1-8 (radio_ready.rhymes:
                 clean words rhyming with the original end word or its rhyme partners,
                 ranked by fit in the line and meaning; computed in the background)

Every verdict is saved to the sheet at once (close it in Excel first). Unvetted rows come
first, the most distinctive rewrites before generic ones ("stuff", "things"): rows are ordered
by how rare the new words are in clean lyrics (data/processed/match/corpus*.jsonl when
present). Kept and dropped sections are both shown; radio_ready.pairs uses the verdicts:
a dislike leaves the section out, and a dropped section whose rows are all liked or edited
comes back.

Choose mode (--choose PATTERN): one screen per section with every version from the matching
generation files (default: all noise styles, "*noise*"); only the changed lines of each
version are shown. Press 1-9 for the best version, 0 for none; Up / Down go back / skip.
Choices go to annotations/noise_choices.csv (id, choice = generation name or "none"), which
`python -m radio_ready.pairs --choices` turns into pairs.

Run:  python -m radio_ready.vet [--all]
      python -m radio_ready.vet --choose "*noise*"
"""

import argparse
import json
import math
import queue
import re
import threading
import tkinter as tk
from collections import Counter
from tkinter import font as tkfont

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.pairs import (
    CHOICES_CSV, EDITS_CSV, SECTION_FILES, fill_text, read_choices, read_edits, write_choices,
    write_edits,
)
from radio_ready.text import rhyme_words, syllables, word_syllables

WORD = re.compile(r"[a-z']+")
CORPUS_FILES = ["corpus_200k.jsonl", "corpus.jsonl"]


def word_rarity() -> tuple[Counter, int]:
    for name in CORPUS_FILES:
        path = paths.PROCESSED / "match" / name
        if path.exists():
            counts: Counter = Counter()
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    counts.update(WORD.findall(json.loads(line)["text"].lower()))
                    if i >= 300_000:
                        break
            return counts, sum(counts.values())
    return Counter(), 1


def uniqueness(original: str, rewrite: str, counts: Counter, total: int) -> float:
    """Mean surprisal of the words the rewrite adds (rare words score high, "stuff" low)."""
    new = [w for w in WORD.findall(rewrite.lower()) if w not in set(WORD.findall(original.lower()))]
    if not new:
        return 0.0
    return sum(-math.log((counts[w] + 1) / total) for w in new) / len(new)


def load_sections(ids: set[str]) -> dict[str, dict]:
    found = {}
    for path in SECTION_FILES.values():
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            m = re.search(r'"id": "([0-9a-f]+)"', line[:60])
            if m and m.group(1) in ids:
                found[m.group(1)] = json.loads(line)
    return found


class App:
    def __init__(self, root, rows, order, sections, detector):
        self.root, self.rows, self.order = root, rows, order
        self.sections, self.detector = sections, detector
        self.pos, self.history, self.context = 0, [], False
        self.suggestions, self.results, self.pending = [], queue.Queue(), None
        base = tkfont.nametofont("TkDefaultFont").actual()["family"]
        big = (base, 15)
        root.title("RadioReady vetting")
        root.geometry("1000x430")
        self.header = tk.Label(root, anchor="w", font=(base, 11))
        self.header.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(root, text="original", anchor="w", fg="#777").pack(fill="x", padx=12)
        self.orig = tk.Text(root, height=1, wrap="word", font=big, relief="flat", bg="#f4f4f4")
        self.orig.pack(fill="x", padx=12)
        self.orig.tag_configure("explicit", foreground="#c0392b", font=(base, 15, "bold"))
        self.orig.tag_configure("current", background="#fff3b0")
        tk.Label(root, text="rewrite (edit in place)", anchor="w", fg="#777").pack(
            fill="x", padx=12, pady=(8, 0))
        self.edit = tk.Text(root, height=1, wrap="word", font=big, undo=True)
        self.edit.pack(fill="x", padx=12)
        self.edit.bind("<KeyRelease>", lambda e: self.update_info())
        self.ctx = tk.Text(root, height=5, wrap="word", font=(base, 12), relief="flat",
                           bg="#f9f9f9")
        self.ctx.tag_configure("current", background="#fff3b0")
        self.info = tk.Label(root, anchor="w", fg="#555", font=(base, 11))
        self.info.pack(fill="x", padx=12, pady=(6, 0))
        self.rhymes = tk.Label(root, anchor="w", fg="#2c6e49", font=(base, 12), justify="left",
                               wraplength=960)
        self.rhymes.pack(fill="x", padx=12, pady=(4, 0))
        bar = tk.Frame(root)
        bar.pack(fill="x", padx=12, pady=10)
        for text, verdict in [("Dislike (Esc)", "dislike"), ("Like (Enter)", "like"),
                              ("Superlike (Ctrl+Enter)", "superlike")]:
            tk.Button(bar, text=text, width=22, takefocus=False,
                      command=lambda v=verdict: self.vote(v)).pack(side="left", padx=(0, 8))
        tk.Button(bar, text="Context (Tab)", takefocus=False,
                  command=self.toggle).pack(side="right")
        tk.Button(bar, text="Skip (Down)", takefocus=False,
                  command=self.skip).pack(side="right", padx=(0, 8))
        tk.Button(bar, text="Back (Up)", takefocus=False,
                  command=self.back).pack(side="right", padx=(0, 8))
        self.status = tk.Label(root, anchor="w", fg="#999")
        self.status.pack(fill="x", padx=12)
        self.edit.bind("<Return>", lambda e: self.vote("like") or "break")
        self.edit.bind("<Control-Return>", lambda e: self.vote("superlike") or "break")
        root.bind("<Escape>", lambda e: self.vote("dislike"))
        self.edit.bind("<Tab>", lambda e: self.toggle() or "break")
        root.bind("<Control-z>", lambda e: self.back() or "break")
        self.edit.bind("<Up>", lambda e: self.back() or "break")
        self.edit.bind("<Down>", lambda e: self.skip() or "break")
        for n in range(1, 9):
            self.edit.bind(f"<Alt-Key-{n}>", lambda e, n=n: self.use_rhyme(n) or "break")
        self.show()
        self.poll()

    def row(self) -> dict:
        return self.rows[self.order[self.pos]]

    def proposed(self, row: dict) -> str:
        return row.get("edit", "").strip() or row.get("rewrite", "") or row["original"]

    def current_lines(self, sid: str) -> list[str]:
        lines = list(self.sections[sid]["lines"]) if sid in self.sections else []
        for r in self.rows:
            if r["id"] == sid and r.get("line", "").isdigit() and int(r["line"]) <= len(lines):
                lines[int(r["line"]) - 1] = self.proposed(r)
        return lines

    def show(self) -> None:
        if self.pos >= len(self.order):
            self.header.config(text="All rows vetted. Close the window.")
            for w in (self.orig, self.edit):
                w.delete("1.0", "end")
            return
        r = self.row()
        done = sum(bool(x.get("verdict")) for x in self.rows)
        self.header.config(text=f'{r["artist"]} - {r["title"]}   line {r["line"]}   '
                                f'[{r.get("status", "")}]   {done}/{len(self.rows)} vetted'
                                + (f'   (was: {r["verdict"]})' if r.get("verdict") else ""))
        self.orig.config(state="normal")
        self.orig.delete("1.0", "end")
        self.orig.insert("1.0", r["original"])
        low = r["original"].lower()
        for w in set(self.detector.find(r["original"])):
            for m in re.finditer(rf"(?<![\w']){re.escape(w)}(?![\w'])", low):
                self.orig.tag_add("explicit", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        self.orig.config(state="disabled")
        self.edit.delete("1.0", "end")
        self.edit.insert("1.0", self.proposed(r))
        self.edit.focus_set()
        self.edit.mark_set("insert", "end-1c")
        self.context = False  # context closes on every line
        self.render_context()
        self.update_info()
        self.load_rhymes()

    def render_context(self) -> None:
        if not self.context:
            self.ctx.pack_forget()
            return
        r = self.row()
        sid, i = r["id"], int(r["line"]) - 1
        orig = self.sections.get(sid, {}).get("lines", [])
        new = self.current_lines(sid)
        start = (i // 4) * 4
        self.ctx.config(state="normal")
        self.ctx.delete("1.0", "end")
        for title, lines in [("original", orig), ("rewrite", new)]:
            self.ctx.insert("end", f"{title}\n")
            for j in range(start, min(len(lines), start + 4)):
                tag = ("current",) if j == i else ()
                self.ctx.insert("end", f"  {lines[j]}\n", tag)
        self.ctx.config(state="disabled", height=11)
        self.ctx.pack(fill="x", padx=12, pady=(8, 0), before=self.info)

    def update_info(self) -> None:
        if self.pos >= len(self.order):
            return
        r, text = self.row(), self.edit.get("1.0", "end").strip()
        left = self.detector.find(text)
        self.info.config(text=f"syllables {syllables(r['original'])} -> {syllables(text)}"
                              + (f"   still explicit: {', '.join(sorted(set(left)))}" if left
                                 else ""))

    def toggle(self) -> None:
        self.context = not self.context
        self.render_context()

    def vote(self, verdict: str) -> None:
        if self.pos >= len(self.order):
            return
        r = self.row()
        text = self.edit.get("1.0", "end").strip()
        r["edit"] = text if text and text != r.get("rewrite", "") else ""
        r["verdict"] = verdict
        try:
            write_edits(self.rows)
            self.status.config(text=f"saved: line {r['line']} {verdict}")
        except PermissionError:
            self.status.config(text="could not save: close the sheet in Excel (kept in memory)")
        self.history.append(self.pos)
        self.pos += 1
        self.show()

    def back(self) -> None:
        if self.history:
            self.pos = self.history.pop()
            self.show()

    def skip(self) -> None:
        if self.pos < len(self.order):
            self.history.append(self.pos)
            self.pos += 1
            self.show()

    # --- rhyme suggestions ----------------------------------------------------------------

    def rhyme_request(self) -> tuple | None:
        r = self.row()
        rec = self.sections.get(r["id"])
        if not rec:
            return None
        i = int(r["line"]) - 1
        scheme = rec["rhyme_scheme"]
        targets = []
        for j, l in enumerate(rec["lines"]):
            if j == i or scheme[j] == scheme[i]:
                found = rhyme_words(l)
                if found and found[-1] not in targets:
                    targets.append(found[-1])
        end = rhyme_words(r["original"])
        n = word_syllables(end[-1]) if end else 1
        before = rec["lines"][i - 1] if i > 0 else ""
        return targets, n, before, sorted(set(self.detector.find(r["original"])))

    def load_rhymes(self) -> None:
        """Queue a suggestion request; one worker thread answers the latest request only."""
        self.suggestions = []
        request = self.rhyme_request()
        line = self.edit.get("1.0", "end").strip()
        if not request or not line:
            self.rhymes.config(text="")
            return
        self.rhymes.config(text="rhymes: ...")
        self.pending = (self.pos, line, request)
        if not getattr(self, "worker", None):
            self.worker = threading.Thread(target=self.work, daemon=True)
            self.worker.start()

    def work(self) -> None:
        from radio_ready.rhymes import suggest
        from radio_ready.scoring import Similarity

        try:
            sim = Similarity(on="cpu")
            embed = lambda t: sim.model.encode(t, normalize_embeddings=True)  # noqa: E731
        except Exception as e:  # noqa: BLE001
            self.results.put((None, [], str(e)))
            return
        while True:
            job, self.pending = self.pending, None
            if job is None:
                threading.Event().wait(0.05)
                continue
            pos, line, (targets, n, before, explicit) = job
            try:
                found, err = suggest(line, targets, n, embed, k=8, before=before,
                                     explicit=explicit), ""
            except Exception as e:  # noqa: BLE001 (shown, never fatal)
                found, err = [], str(e)
            self.results.put((pos, found, err))  # tkinter is only touched by the main thread

    def poll(self) -> None:
        while not self.results.empty():
            self.show_rhymes(*self.results.get())
        self.root.after(100, self.poll)

    def show_rhymes(self, pos, found: list[dict], err: str) -> None:
        if pos is not None and pos != self.pos:
            return  # moved on
        self.suggestions = found
        if err:
            self.rhymes.config(text=f"rhymes unavailable: {err}")
        elif not found:
            self.rhymes.config(text="rhymes: none found")
        else:
            self.rhymes.config(text="rhymes (Alt+n):  " + "   ".join(
                f"{k}={s['word']}" for k, s in enumerate(found, start=1)))

    def use_rhyme(self, n: int) -> None:
        if n <= len(self.suggestions):
            from radio_ready.rhymes import replace_end

            text = self.edit.get("1.0", "end").strip()
            self.edit.delete("1.0", "end")
            self.edit.insert("1.0", replace_end(text, self.suggestions[n - 1]["word"]))
            self.update_info()


class ChooseApp:
    """Pick the best of several versions of a section (or none)."""

    def __init__(self, root, items, choices, sections, detector):
        self.root, self.items, self.choices = root, items, choices
        self.sections, self.detector = sections, detector
        self.pos, self.history = 0, []
        base = tkfont.nametofont("TkDefaultFont").actual()["family"]
        root.title("RadioReady: choose the best version")
        root.geometry("1100x650")
        self.header = tk.Label(root, anchor="w", font=(base, 11))
        self.header.pack(fill="x", padx=12, pady=(10, 4))
        self.text = tk.Text(root, wrap="word", font=(base, 13), relief="flat", bg="#fafafa")
        self.text.pack(fill="both", expand=True, padx=12)
        self.text.tag_configure("explicit", foreground="#c0392b", font=(base, 13, "bold"))
        self.text.tag_configure("label", foreground="#2c6e49", font=(base, 13, "bold"))
        self.text.tag_configure("muted", foreground="#999")
        self.status = tk.Label(root, anchor="w", fg="#777",
                               text="1-9 choose   0 none   Up back   Down skip")
        self.status.pack(fill="x", padx=12, pady=8)
        for n in range(10):
            root.bind(str(n), lambda e, n=n: self.choose(n))
        root.bind("<Up>", lambda e: self.back())
        root.bind("<Down>", lambda e: self.skip())
        self.show()

    def show(self) -> None:
        t = self.text
        t.config(state="normal")
        t.delete("1.0", "end")
        if self.pos >= len(self.items):
            self.header.config(text="All sections chosen. Close the window.")
            t.config(state="disabled")
            return
        sid, versions = self.items[self.pos]
        rec = self.sections[sid]
        done = sum(1 for s, _ in self.items if s in self.choices)
        was = self.choices.get(sid)
        self.header.config(text=f'{rec["artist"]} - {rec["title"]} ({rec.get("section_header")})'
                                f'   {done}/{len(self.items)} chosen'
                                + (f"   (was: {was})" if was else ""))
        t.insert("end", "original\n", "label")
        for i, line in enumerate(rec["lines"]):
            start = t.index("end-1c")
            t.insert("end", f"  {i + 1}. {line}\n")
            for w in set(self.detector.find(line)):
                for m in re.finditer(rf"(?<![\w']){re.escape(w)}(?![\w'])", line.lower()):
                    off = len(f"  {i + 1}. ") + m.start()
                    t.tag_add("explicit", f"{start}+{off}c", f"{start}+{off + len(w)}c")
        for k, (name, lines) in enumerate(versions, start=1):
            t.insert("end", f"\n{k}\n", "label")
            changed = [(i, l) for i, (l, o) in enumerate(zip(lines, rec["lines"])) if l != o]
            for i, l in changed:
                t.insert("end", f"  {i + 1}. {l}\n")
            if not changed:
                t.insert("end", "  (no change)\n", "muted")
        t.config(state="disabled")

    def choose(self, n: int) -> None:
        if self.pos >= len(self.items):
            return
        sid, versions = self.items[self.pos]
        if n and n > len(versions):
            return
        self.choices[sid] = "none" if n == 0 else versions[n - 1][0]
        try:
            write_choices(self.choices)
            self.status.config(text=f"saved: {self.choices[sid]}")
        except PermissionError:
            self.status.config(text="could not save: close the sheet in Excel (kept in memory)")
        self.history.append(self.pos)
        self.pos += 1
        self.show()

    def back(self) -> None:
        if self.history:
            self.pos = self.history.pop()
            self.show()

    def skip(self) -> None:
        if self.pos < len(self.items):
            self.history.append(self.pos)
            self.pos += 1
            self.show()


def choose_items(pattern: str, show_all: bool, choices: dict) -> list[tuple[str, list]]:
    """[(section id, [(generation name, lines), ...])] with distinct valid versions."""
    by_id: dict[str, list] = {}
    for path in sorted(paths.GENERATIONS_DIR.glob(f"{pattern}.jsonl")):
        for line in open(path, encoding="utf-8"):
            g = json.loads(line)
            if g.get("valid") and g.get("lines"):
                found = by_id.setdefault(g["id"], [])
                if all(lines != g["lines"] for _, lines in found):
                    found.append((path.stem, g["lines"]))
    return [(sid, v[:9]) for sid, v in by_id.items() if show_all or sid not in choices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vet synthetic rewrites line by line.")
    parser.add_argument("--all", action="store_true", help="also show rows already vetted")
    parser.add_argument("--choose", default=None, metavar="PATTERN",
                        help='choose the best version per section, e.g. "*noise*"')
    args = parser.parse_args()

    if args.choose:
        choices = read_choices()
        items = choose_items(args.choose, args.all, choices)
        if not items:
            raise SystemExit(f"nothing to choose for {args.choose!r}")
        root = tk.Tk()
        ChooseApp(root, items, choices, load_sections({s for s, _ in items}), ExplicitDetector())
        root.mainloop()
        return

    rows = read_edits()
    if not rows:
        raise SystemExit(f"no rows in {EDITS_CSV}; run radio_ready.pairs first")
    sections = load_sections({r["id"] for r in rows})
    fill_text(rows, lambda sid: sections.get(sid, {}).get("lines"))
    counts, total = word_rarity()
    scored = [(uniqueness(r["original"], r.get("edit") or r.get("rewrite", ""), counts, total), k)
              for k, r in enumerate(rows) if args.all or not r.get("verdict")]
    order = [k for _, k in sorted(scored, reverse=True)]
    root = tk.Tk()
    App(root, rows, order, sections, ExplicitDetector())
    root.mainloop()


if __name__ == "__main__":
    main()
