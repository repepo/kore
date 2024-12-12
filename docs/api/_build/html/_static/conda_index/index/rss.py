"""
Generate RSS feed from channeldata.
"""
from __future__ import annotations

import time
from xml.dom import minidom


# Should we have either n packages or m days?
def get_recent_packages(channeldata, threshold_packages=100) -> list[tuple[str, dict]]:
    return sorted(
        channeldata["packages"].items(),
        key=lambda item: item[1].get("timestamp", 0),
        reverse=True,
    )[:threshold_packages]


def _iso822(timestamp):
    return time.strftime("%a, %d %b %Y %T GMT", time.gmtime(timestamp))


def _get_channel(channel_name, packages):
    return {
        "title": f"anaconda.org/{channel_name}",
        "link": f"https://conda.anaconda.org/{channel_name}",
        "description": f"The most recent {len(packages)} updates for {channel_name}.",
        "pubDate": _iso822(time.time()),
        "lastBuildDate": _iso822(time.time()),
    }


def _get_title(name, version, subdirs):
    return f"{name} {version} [{', '.join(sorted({x for x in subdirs}))}]"


def _get_items(packages: list[tuple[str, dict]]):

    items = []
    for name, package in packages:

        def __(x):
            return package.get(x)

        def coalesce(*args, default="No description."):
            for arg in [a for a in args if __(a)]:
                return package[arg]
            return default

        item = {
            # Example: "7zip 19.00 [osx-64, win-64]"
            "title": _get_title(name, __("version"), __("subdirs")),
            "description": coalesce("description", "summary"),
            "link": __("doc_url"),  # URI - project or project docs
            "comments": __("dev_url"),  # URI
            "guid": __("source_url"),  # URI - download link
            "pubDate": _iso822(__("timestamp")),
            "source": __("home"),  # URI
        }
        items.append({k: v for k, v in item.items() if v})
    return items


def get_rss(channel_name, channeldata):
    newdoc: minidom.Document = minidom.parseString("<rss version='2.0'></rss>")

    def append_strings(node: minidom.Element, strings: dict[str, str]):
        for key, value in strings.items():
            e: minidom.Element = newdoc.createElement(key)
            e.appendChild(newdoc.createTextNode(str(value)))
            node.appendChild(e)

    packages = get_recent_packages(channeldata)

    channel: minidom.Element = newdoc.createElement("channel")
    append_strings(channel, _get_channel(channel_name, packages))

    for package in _get_items(packages):
        item = newdoc.createElement("item")
        append_strings(item, package)
        channel.appendChild(item)

    rss: minidom.Element = newdoc.documentElement
    rss.appendChild(channel)

    return newdoc.toprettyxml(indent="  ")


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    channel, channeldata_fn, threshold_days = sys.argv[1:]
    with open(channeldata_fn) as fd:
        print(get_rss(channel, json.load(fd)))
