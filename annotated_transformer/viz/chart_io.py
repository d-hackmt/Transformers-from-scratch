"""Save an Altair chart to disk (HTML always, PNG when possible)."""

import os


def save_chart(chart, path_without_ext: str) -> list:
    """Write ``chart`` to ``<path_without_ext>.html`` (and ``.png`` if available).

    HTML export only needs Altair itself.  PNG export needs ``vl-convert-python``
    (or another Altair image backend); if that is not installed the PNG is
    skipped with a warning rather than raising.  Returns the list of files
    written.
    """
    os.makedirs(os.path.dirname(path_without_ext) or ".", exist_ok=True)
    written = []

    html_path = path_without_ext + ".html"
    chart.save(html_path)
    written.append(html_path)

    png_path = path_without_ext + ".png"
    try:
        chart.save(png_path)
        written.append(png_path)
    except Exception as exc:  # missing image backend, etc.
        print(f"(skipped PNG for {png_path}: {exc})")

    return written
