"""Minimal PDF content-stream interpreter that extracts painted vector paths
(with absolute page coordinates, paint mode and colour) and text runs with
their positions.  Used to recover plotted data points from vector figures.

Only the subset of the PDF graphics model needed for matplotlib/Origin
figures is implemented: q/Q, cm, m/l/c/v/y/h/re, path painting operators,
colour operators, Form XObjects (recursively, with their /Matrix), and the
text operators BT/ET/Tf/Td/TD/Tm/T*/Tj/TJ/'/" (text positions only).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream, NameObject


def mat_mul(a, b):
    """Return a*b for PDF 3x3 matrices stored as 6-tuples (a b c d e f)."""
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return (a0 * b0 + a1 * b2, a0 * b1 + a1 * b3,
            a2 * b0 + a3 * b2, a2 * b1 + a3 * b3,
            a4 * b0 + a5 * b2 + b4, a4 * b1 + a5 * b3 + b5)


def apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


@dataclass
class Path:
    subpaths: List[np.ndarray]          # each (n,2) array of page coords
    closed: List[bool]
    fill: bool
    stroke: bool
    fill_color: Tuple[float, ...]
    stroke_color: Tuple[float, ...]
    linewidth: float
    depth: int
    page: int

    @property
    def points(self):
        return np.vstack(self.subpaths) if self.subpaths else np.zeros((0, 2))

    @property
    def bbox(self):
        p = self.points
        return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())

    @property
    def centroid(self):
        p = self.points
        return (p[:, 0].mean(), p[:, 1].mean())

    @property
    def size(self):
        x0, y0, x1, y1 = self.bbox
        return max(x1 - x0, y1 - y0)


@dataclass
class TextRun:
    text: str
    x: float
    y: float
    size: float
    page: int


@dataclass
class State:
    ctm: tuple
    fill_color: tuple = (0.0,)
    stroke_color: tuple = (0.0,)
    linewidth: float = 1.0


def _num(a):
    return float(a)


class Interpreter:
    def __init__(self, reader: PdfReader):
        self.reader = reader
        self.paths: List[Path] = []
        self.texts: List[TextRun] = []

    def run_page(self, pageno: int):
        page = self.reader.pages[pageno]
        content = page.get_contents()
        if content is None:
            return
        cs = ContentStream(content, self.reader)
        base = (1, 0, 0, 1, 0, 0)
        self._exec(cs.operations, page.get("/Resources"), State(base), pageno, 0)

    def _exec(self, ops, resources, st: State, pageno, depth):
        stack: List[State] = []
        cur = []          # current subpath points (list of (x,y) in user space)
        subpaths = []     # finished subpaths (np arrays in page coords)
        closed = []
        start = None
        # text state
        tm = None
        tlm = None
        tfs = 1.0
        for args, op in ops:
            if op == b"q":
                stack.append(State(st.ctm, st.fill_color, st.stroke_color, st.linewidth))
            elif op == b"Q":
                if stack:
                    st = stack.pop()
            elif op == b"cm":
                m = tuple(_num(a) for a in args)
                st.ctm = mat_mul(m, st.ctm)
            elif op == b"w":
                st.linewidth = _num(args[0])
            elif op in (b"g", b"rg", b"k", b"sc", b"scn"):
                try:
                    st.fill_color = tuple(_num(a) for a in args if not isinstance(a, NameObject))
                except Exception:
                    pass
            elif op in (b"G", b"RG", b"K", b"SC", b"SCN"):
                try:
                    st.stroke_color = tuple(_num(a) for a in args if not isinstance(a, NameObject))
                except Exception:
                    pass
            # ---- path construction
            elif op == b"m":
                if cur:
                    subpaths.append(np.array([apply(st.ctm, *p) for p in cur])); closed.append(False)
                cur = [(_num(args[0]), _num(args[1]))]
                start = cur[0]
            elif op == b"l":
                cur.append((_num(args[0]), _num(args[1])))
            elif op in (b"c", b"v", b"y"):
                vals = [_num(a) for a in args]
                if op == b"c":
                    p1, p2, p3 = (vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])
                elif op == b"v":
                    p1 = cur[-1] if cur else (vals[0], vals[1]); p2, p3 = (vals[0], vals[1]), (vals[2], vals[3])
                else:
                    p1, p2, p3 = (vals[0], vals[1]), (vals[2], vals[3]), (vals[2], vals[3])
                p0 = cur[-1] if cur else p1
                for t in (0.25, 0.5, 0.75, 1.0):
                    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
                    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
                    cur.append((x, y))
            elif op == b"h":
                if cur:
                    subpaths.append(np.array([apply(st.ctm, *p) for p in cur])); closed.append(True)
                    cur = []
            elif op == b"re":
                x, y, w, h = (_num(a) for a in args)
                if cur:
                    subpaths.append(np.array([apply(st.ctm, *p) for p in cur])); closed.append(False); cur = []
                pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                subpaths.append(np.array([apply(st.ctm, *p) for p in pts])); closed.append(True)
            # ---- path painting
            elif op in (b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"S", b"s", b"n"):
                if cur:
                    subpaths.append(np.array([apply(st.ctm, *p) for p in cur]))
                    closed.append(op in (b"s", b"b", b"b*"))
                    cur = []
                if subpaths and op != b"n":
                    fill = op in (b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*")
                    stroke = op in (b"S", b"s", b"B", b"B*", b"b", b"b*")
                    lw = st.linewidth * math.sqrt(abs(st.ctm[0] * st.ctm[3] - st.ctm[1] * st.ctm[2]))
                    self.paths.append(Path(subpaths, closed, fill, stroke, st.fill_color, st.stroke_color, lw, depth, pageno))
                subpaths, closed = [], []
            elif op in (b"W", b"W*"):
                pass
            # ---- XObjects
            elif op == b"Do":
                name = args[0]
                xobjs = resources.get("/XObject") if resources else None
                if xobjs is None:
                    continue
                xo = xobjs.get_object().get(name)
                if xo is None:
                    continue
                xo = xo.get_object()
                if xo.get("/Subtype") == "/Form":
                    m = xo.get("/Matrix", [1, 0, 0, 1, 0, 0])
                    sub = State(mat_mul(tuple(_num(a) for a in m), st.ctm), st.fill_color, st.stroke_color, st.linewidth)
                    subops = ContentStream(xo, self.reader).operations
                    self._exec(subops, xo.get("/Resources", resources), sub, pageno, depth + 1)
            # ---- text (positions only)
            elif op == b"BT":
                tm = (1, 0, 0, 1, 0, 0); tlm = tm
            elif op == b"ET":
                tm = None
            elif op == b"Tf":
                tfs = _num(args[1])
            elif op == b"Tm":
                tm = tuple(_num(a) for a in args); tlm = tm
            elif op in (b"Td", b"TD"):
                if tlm is not None:
                    tlm = mat_mul((1, 0, 0, 1, _num(args[0]), _num(args[1])), tlm); tm = tlm
            elif op == b"T*":
                pass
            elif op in (b"Tj", b"TJ", b"'", b'"'):
                if tm is None:
                    continue
                if op == b"TJ":
                    s = "".join(str(a) for a in args[0] if not isinstance(a, (int, float)) and hasattr(a, "original_bytes") or isinstance(a, str))
                    try:
                        s = "".join(a.decode("latin-1") if isinstance(a, bytes) else (a if isinstance(a, str) else "") for a in [x.original_bytes if hasattr(x, "original_bytes") else x for x in args[0]])
                    except Exception:
                        pass
                else:
                    a = args[-1] if op != b'"' else args[-1]
                    s = a.original_bytes.decode("latin-1") if hasattr(a, "original_bytes") else str(a)
                full = mat_mul(tm, st.ctm)
                x, y = full[4], full[5]
                size = tfs * math.sqrt(abs(full[0] * full[3] - full[1] * full[2]))
                self.texts.append(TextRun(s, x, y, size, pageno))
        if cur:
            pass
        return


def extract(pdf_path: str, pages=None):
    r = PdfReader(pdf_path)
    it = Interpreter(r)
    for i in (pages if pages is not None else range(len(r.pages))):
        it.run_page(i)
    return it


if __name__ == "__main__":
    import sys
    it = extract(sys.argv[1], [int(sys.argv[2])])
    print(len(it.paths), "paths;", len(it.texts), "text runs")
    for t in it.texts[:50]:
        print(f"text {t.x:8.2f} {t.y:8.2f} size {t.size:5.2f} {t.text!r}")
