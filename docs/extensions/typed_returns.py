import re
from typing import Any, Iterable, Iterator, List

from sphinx.application import Sphinx
from sphinx.ext.napoleon import NumpyDocstring


def process_return(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        m = re.fullmatch(r"(?P<param>\w+)\s+:\s+(?P<type>[\w.]+)", line)
        if m:
            # Once this is in scanpydoc, we can use the fancy hover stuff
            if m["param"]:
                yield f'**{m["param"]}** : :class:`~{m["type"]}`'
            else:
                yield f':class:`~{m["type"]}`'
        else:
            yield line


def scanpy_parse_returns_section(self: NumpyDocstring, _: Any) -> List[str]:
    lines_raw = list(process_return(self._dedent(self._consume_to_next_section())))
    lines = self._format_block(":returns: ", lines_raw)
    if lines and lines[-1]:
        lines.append("")
    return lines


def setup(app: Sphinx) -> None:
    NumpyDocstring._parse_returns_section = scanpy_parse_returns_section
