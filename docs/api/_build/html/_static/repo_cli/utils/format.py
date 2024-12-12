from __future__ import unicode_literals

import argparse

INITIAL_SPACE = "     "
fmt_package_headers = (
    INITIAL_SPACE + "%(channel_path)-30s | %(name)-40s | %(version)8s | %(family)-12s "
    "| %(build_number)-10s | %(license)-15s | %(subdirs)-15s"
)

fmt_package_simple_headers = (
    INITIAL_SPACE + "%(name)-40s | %(file_count)-12s | %(download_count)-13s "
    "| %(license)-15s"
)
fmt_channel_headers = (
    INITIAL_SPACE
    + "%(channel_path)-30s | %(privacy)-10s | %(owners)15s | %(artifact_count)-12s "
    "| %(download_count)-11s | %(subchannel_count)-14s |  %(mirror_count)-9s | %(description)-30s"
)
fmt_mirror_header_spacer = {
    "id": "-" * 36,
    "name": "-" * 15,
    "type": "-" * 8,
    "mode": "-" * 10,
    "state": "-" * 10,
    "source_root": "-" * 50,
    "proxy": "-" * 40,
    "last_run_at": "-" * 30,
    "updated_at": "-" * 30,
}
fmt_mirror_headers = (
    INITIAL_SPACE + "%(id)-36s | %(name)-15s | %(type)8s | %(mode)-10s "
    "| %(state)-10s | %(source_root)-50s | %(proxy)-40s | %(last_run_at)-30s | %(updated_at)-30s"
)

fmt_all_mirror_header_spacer = {
    "id": "-" * 36,
    "name": "-" * 15,
    "source_root": "-" * 45,
    "type": "-" * 18,
    "privacy": "-" * 14,
    "channel": "-" * 18,
    "mode": "-" * 10,
    "state": "-" * 10,
    "last_run_at": "-" * 34,
    "created_at": "-" * 34,
    "updated_at": "-" * 34,
}

fmt_all_mirror_headers = (
    INITIAL_SPACE
    + "%(id)-36s | %(name)-15s | %(source_root)45s | %(type)-18s | %(channel)-18s  | %(privacy)-14s  "
    "| %(mode)-10s | %(state)-10s | %(last_run_at)-34s | %(created_at)-34s | %(updated_at)-34s"
)

fmt_sync_state_header_spacer = {
    "active": "-" * 10,
    "failed": "-" * 10,
    "passive": "-" * 10,
    "removed": "-" * 10,
    "last_ckey": "-" * 60,
    "last_exception": "-" * 18,
    "total_steps_count": "-" * 18,
    "current_step_number": "-" * 20,
    "current_step_percentage": "-" * 25,
    "current_step_description": "-" * 34,
}
fmt_sync_state_headers = (
    INITIAL_SPACE
    + "%(active)-10s | %(failed)-10s | %(passive)-10s | %(removed)-10s | %(last_ckey)-60s  | %(last_exception)-18s  "
    "| %(total_steps_count)-18s | %(current_step_number)-20s | %(current_step_percentage)-25s | %(current_step_description)-34s"
)

fmt_sync_state_mirror_filtered_header_spacer = {
    "cve": "-" * 18,
    "date": "-" * 18,
    "license": "-" * 18,
    "only_specs": "-" * 18,
    "exclude_specs": "-" * 18,
    "duplicated_legacy_conda": "-" * 18,
}
fmt_sync_state_mirror_filtered_header = (
    INITIAL_SPACE
    + "%(cve)-18s | %(date)-18s | %(license)-18s | %(only_specs)-18s | %(exclude_specs)-18s  | %(duplicated_legacy_conda)-18s"
)

fmt_sync_state_channel_filtered_header_spacer = {
    "license": "-" * 18,
    "exclude_specs": "-" * 18,
}
fmt_sync_state_channel_filtered_header = (
    INITIAL_SPACE + "%(license)-18s | %(exclude_specs)-18s"
)


class PackagesFormatter:
    def __init__(self, log):
        self.log = log

    @staticmethod
    def format_package_header(simple=False):
        if simple:
            package_header = {
                "name": "Name",
                "file_count": "# of files",
                "download_count": "# of download",
                "license": "License",
            }
            fmt = fmt_package_simple_headers
        else:
            package_header = {
                "channel_path": "Channel",
                "name": "Name",
                "family": "Family",
                "version": "Version",
                "subdirs": "Platforms",
                "license": "License",
                "build_number": "Build",
            }
            fmt = fmt_package_headers
        return fmt % package_header

    def log_format_package_header(self, simple=False):
        self.log.info(self.format_package_header(simple=simple))

    @staticmethod
    def format_package(package, simple=False):
        package = package.copy()
        package.update(package["metadata"])

        if package["subchannel"]:
            package["channel_path"] = "%s/%s" % (
                package["channel"],
                package["subchannel"],
            )
        else:
            package["channel_path"] = package["channel"]

        if "subdirs" not in package:
            package["subdirs"] = []

        if "build_number" not in package:
            package["build_number"] = ""

        if "license" not in package:
            package["license"] = ""

        package["full_name"] = "%s::%s" % (package["channel_path"], package["name"])
        package["subdirs"] = ", ".join(
            str(x) for x in package["subdirs"] if x is not None
        )

        if simple:
            fmt = fmt_package_simple_headers
        else:
            fmt = fmt_package_headers

        return fmt % package

    def log_format_package(self, package, simple=False):
        self.log.info(self.format_package(package, simple=simple))

    @staticmethod
    def format_channel_header():
        package_header = {
            "channel_path": "Channel",
            "privacy": "Privacy",
            "owners": "Owners",
            "artifact_count": "# Artifacts",
            "download_count": "# Downloads",
            "subchannel_count": "# Subchannels",
            "mirror_count": "# Mirrors",
            "description": "Description",
        }
        return fmt_channel_headers % package_header

    def log_format_channel_header(self):
        self.log.info(self.format_channel_header())

    @staticmethod
    def format_channel(channel):
        channel = channel.copy()

        if channel["parent"]:
            channel["channel_path"] = "%s/%s" % (channel["parent"], channel["name"])
        else:
            channel["channel_path"] = channel["name"]

        channel["owners"] = ", ".join(
            str(x) for x in channel["owners"] if x is not None
        )
        return fmt_channel_headers % channel

    def log_format_channel(self, channel):
        self.log.info(self.format_channel(channel))

    def format(self, packages, metadata, simple=False):
        self.log_format_package_header(simple=simple)

        if simple:
            package_header = {
                "name": "-" * 40,
                "file_count": "-" * 8,
                "download_count": "-" * 13,
                "license": "-" * 15,
            }

            self.log.info(fmt_package_simple_headers % package_header)
        else:
            package_header = {
                "channel_path": "-" * 30,
                "name": "-" * 40,
                "version": "-" * 8,
                "family": "-" * 12,
                "subdirs": "-" * 15,
                "license": "-" * 15,
                "build_number": "-" * 10,
            }

            self.log.info(fmt_package_headers % package_header)

        for package in packages:
            self.log_format_package(package, simple=simple)

        if packages:
            end_set = len(packages) + metadata["offset"] if metadata["offset"] else 0
            self.log.info(
                "\n%s%i packages found." % (INITIAL_SPACE, metadata["total_count"])
            )
            self.log.info(
                "%sVisualizing %i-%i interval."
                % (INITIAL_SPACE, len(packages), end_set)
            )
        else:
            self.log.info("No packages found")

        self.log.info("")


class MirrorFormatter:
    keymap = {
        "created_at": "created at",
        "updated_at": "Updated at",
        "last_run_at": "Last run at",
    }

    @staticmethod
    def format_detail(mirror):
        mirror = {key.replace("mirror_", ""): val for key, val in mirror.items()}
        keymap = {
            "created_at": "created at",
            "updated_at": "Updated at",
            "last_run_at": "Last run at",
        }
        mirror_ = dict(mirror)
        resp = [INITIAL_SPACE + "Mirror Details:", INITIAL_SPACE + "---------------"]

        fields = [
            "id",
            "name",
            "type",
            "mode",
            "state",
            "source_root",
            "last_run_at",
            "updated_at",
            "created",
            "cron",
            "proxy",
            "filters",
        ]
        for key in fields:
            label = keymap.get(key, key.replace("_", " "))
            value = mirror_.get(key, "")
            if key == "filters":
                resp.append("%s%s:" % (INITIAL_SPACE, label))
                if value:
                    for filter_key in sorted(value.keys()):
                        resp.append(
                            "%s   %s| %s"
                            % (INITIAL_SPACE, filter_key.ljust(25), value[filter_key])
                        )
            else:
                resp.append("%s%s: %s" % (INITIAL_SPACE, label, value))

        return "\n".join(resp)

    @classmethod
    def format_list_headers(cls):
        mirror_headers = {k: k.capitalize() for k in fmt_mirror_header_spacer}
        mirror_headers.update(cls.keymap)
        return fmt_mirror_headers % mirror_headers

    @staticmethod
    def format_list_item(mirror):
        mirror = {
            key.replace("mirror_", ""): (val if val is not None else "")
            for key, val in mirror.items()
        }
        return fmt_mirror_headers % mirror

    @staticmethod
    def format_list(mirrors):
        lines = []
        lines.append(MirrorFormatter.format_list_headers())
        lines.append(fmt_mirror_headers % fmt_mirror_header_spacer)

        for mirror in mirrors:
            lines.append(MirrorFormatter.format_list_item(mirror))

        return "\n".join(lines)


class AllMirrorFormatter:
    keymap = {
        "source_root": "root source",
        "last_run_at": "last run at",
        "created_at": "created at",
        "updated_at": "updated at",
    }

    @staticmethod
    def format_list_all(mirrors):
        lines = []
        lines.append(AllMirrorFormatter.format_list_all_headers())
        lines.append(fmt_all_mirror_headers % fmt_all_mirror_header_spacer)

        for mirror in mirrors:
            lines.append(AllMirrorFormatter.format_list_item(mirror))

        return "\n".join(lines)

    @classmethod
    def format_list_all_headers(cls):
        mirror_headers = {k: k.capitalize() for k in fmt_all_mirror_header_spacer}
        mirror_headers.update(cls.keymap)
        return fmt_all_mirror_headers % mirror_headers

    @staticmethod
    def format_list_item(mirror):
        mirror.pop("sync_state", None)

        channel = mirror["channel"]
        mirror.pop("channel", None)
        channel["channel"] = channel.pop("name")
        mirror = {**mirror, **channel}

        return fmt_all_mirror_headers % mirror


class SyncStateFormatter:
    keymap = {
        "last_ckey": "Last ckey",
        "last_exception": "Last exception",
        "total_steps_count": "Total steps count",
        "current_step_number": "Current step number",
        "current_step_percentage": "Current step percentage",
        "current_step_description": "Current step description",
    }
    keymap_mirror = {
        "only_specs": "Only specs",
        "exclude_specs": "exclude specs",
        "duplicated_legacy_conda": "Duplicated legacy conda",
    }
    keymap_channel = {"exclude_specs": "Exclude specs"}

    @staticmethod
    def format_sync_state(data, mirror_name):
        mirror = None
        for m in data:
            if m["name"] == mirror_name:
                mirror = m

        if not mirror:
            lines = []
            lines.append("No Mirror named %s" % mirror_name)
            return "\n".join(lines)

        sync_state = mirror["sync_state"]

        lines = []

        lines.append(INITIAL_SPACE + "SYNC STATE:")
        lines.append(SyncStateFormatter.format_sync_state_headers())
        lines.append(fmt_sync_state_headers % fmt_sync_state_header_spacer)
        lines.append(SyncStateFormatter.format_sync_state_general(sync_state))
        lines.append("")

        lines.append(INITIAL_SPACE + "FILTERED PACKAGES BY MIRROR:")
        lines.append(SyncStateFormatter.format_filtered_mirror_headers())
        lines.append(
            fmt_sync_state_mirror_filtered_header
            % fmt_sync_state_mirror_filtered_header_spacer
        )
        lines.append(
            SyncStateFormatter.format_sync_state_filtered_mirror(sync_state["count"])
        )
        lines.append("")

        lines.append(INITIAL_SPACE + "FILTERED PACKAGES BY CHANNEL:")
        lines.append(SyncStateFormatter.format_filtered_channel_headers())
        lines.append(
            fmt_sync_state_channel_filtered_header
            % fmt_sync_state_channel_filtered_header_spacer
        )
        lines.append(
            SyncStateFormatter.format_sync_state_filtered_channel(sync_state["count"])
        )
        lines.append("")

        return "\n".join(lines)

    @classmethod
    def format_sync_state_headers(cls):
        sync_state_headers = {k: k.capitalize() for k in fmt_sync_state_header_spacer}
        sync_state_headers.update(cls.keymap)
        return fmt_sync_state_headers % sync_state_headers

    @staticmethod
    def format_sync_state_general(sync_state):
        sync_state = sync_state.copy()
        count = sync_state["count"]
        sync_state.pop("count", None)
        sync_state.update(count)
        sync_state.pop("mirror_filtered", None)
        sync_state.pop("channel_filtered", None)

        return fmt_sync_state_headers % sync_state

    @classmethod
    def format_filtered_mirror_headers(cls):
        sync_state_mirror_filtered_header = {
            k: k.capitalize() for k in fmt_sync_state_mirror_filtered_header_spacer
        }
        sync_state_mirror_filtered_header.update(cls.keymap_mirror)
        return fmt_sync_state_mirror_filtered_header % sync_state_mirror_filtered_header

    @staticmethod
    def format_sync_state_filtered_mirror(sync_state):
        mirror_filtered = sync_state["mirror_filtered"]
        return fmt_sync_state_mirror_filtered_header % mirror_filtered

    @classmethod
    def format_filtered_channel_headers(cls):
        sync_state_channel_filtered_header = {
            k: k.capitalize() for k in fmt_sync_state_channel_filtered_header_spacer
        }
        sync_state_channel_filtered_header.update(cls.keymap_channel)
        return (
            fmt_sync_state_channel_filtered_header % sync_state_channel_filtered_header
        )

    @staticmethod
    def format_sync_state_filtered_channel(sync_state):
        channel_filtered = sync_state["channel_filtered"]
        return fmt_sync_state_channel_filtered_header % channel_filtered


class FormatterBase:
    entity = ""
    fmt_header_spacer = {}
    fmt_headers = INITIAL_SPACE + ""

    keymap = {
        "date_added": "Published",
        "updated_at": "Updated at",
        "last_run_at": "Last run at",
    }

    fields = []

    @classmethod
    def format_detail(cls, item):
        item_ = cls.normalize_item(item)
        resp = [
            INITIAL_SPACE + "%s Details:" % cls.entity,
            INITIAL_SPACE + "---------------",
        ]

        for key in cls.fields:
            label = cls.keymap.get(key, key).capitalize()
            value = nested_get(item_, key, "")
            resp.append("%s%s: %s" % (INITIAL_SPACE, label, value))

        return "\n".join(resp)

    @classmethod
    def format_list_headers(cls):
        list_headers = {k: k.capitalize() for k in cls.fmt_header_spacer}
        list_headers.update(cls.keymap)
        return cls.fmt_headers % list_headers

    @classmethod
    def format_list_item(cls, item):
        item_ = cls.normalize_item(item)
        return cls.fmt_headers % item_

    @staticmethod
    def normalize_item(item):
        raise NotImplementedError

    @classmethod
    def format_list(cls, items):
        lines = []
        lines.append(cls.format_list_headers())
        lines.append(cls.fmt_headers % cls.fmt_header_spacer)

        for item in items:
            lines.append(cls.format_list_item(item))

        return "\n".join(lines)


class CVEFormatter(FormatterBase):
    entity = "CVE"
    fmt_header_spacer = {
        "id": "-" * 14,
        "score": "-" * 6,
        "score_type": "-" * 10,
        "curated": "-" * 8,
        "packages_count": "-" * 10,
        "description": "-" * 100,
    }
    fmt_headers = (
        INITIAL_SPACE
        + "%(id)-14s | %(score)6s | %(score_type)10s | %(curated)8s | %(packages_count)10s | %(description)-50s"
    )

    keymap = {
        "id": "CVE ID",
        "score_type": "Type",
        "packages_count": "# Packages",
        "cvssv2.cvssV2.accessComplexity": "CVSS V2 Access Complexity",
        "cvssv2.cvssV2.accessVector": "CVSS V2 Access Vector",
        "cvssv2.cvssV2.authentication": "CVSS V2 Authentication",
        "cvssv2.cvssV2.availabilityImpact": "CVSS V2 Availablity Impact",
        "cvssv2.cvssV2.confidentialityImpact": "CVSS V2 Confidentiality Impact",
        "cvssv2.cvssV2.integrityImpact": "CVSS V2 Integrity Impact",
        "cvssv2.cvssV2.baseScore": "CVSS V2 Base Score",
        "cvssv2.severity": "CVSS V2 Severity",
        "cvssv3.cvssV3.attackComplexity": "CVSS V3 Attack Complexity",
        "cvssv3.cvssV3.attackVector": "CVSS V3 Attack Vector",
        "cvssv3.cvssV3.availabilityImpact": "CVSS V3 Availability Impact",
        "cvssv3.cvssV3.baseScore": "CVSS V3 Base Score",
        "cvssv3.cvssV3.baseSeverity": "CVSS V3 Base Severity",
        "cvssv3.cvssV3.confidentialityImpact": "CVSS V3 Confidentiality Impact",
        "cvssv3.cvssV3.integrityImpact": "CVSS V3 Integrity Impact",
        "cvssv3.cvssV3.privilegesRequired": "CVSS V3 Privileges Required",
        "cvssv3.cvssV3.scope": "CVSS V3 Scope",
        "cvssv3.cvssV3.userInteraction": "CVSS V3 User Interaction",
    }

    fields = [
        "id",
        "curated",
        "score",
        "score_type",
        "description",
        "cvssv2.cvssV2.accessComplexity",
        "cvssv2.cvssV2.accessVector",
        "cvssv2.cvssV2.authentication",
        "cvssv2.cvssV2.availabilityImpact",
        "cvssv2.cvssV2.confidentialityImpact",
        "cvssv2.cvssV2.integrityImpact",
        "cvssv2.cvssV2.baseScore",
        "cvssv2.severity",
        "cvssv3.cvssV3.attackComplexity",
        "cvssv3.cvssV3.attackVector",
        "cvssv3.cvssV3.availabilityImpact",
        "cvssv3.cvssV3.baseScore",
        "cvssv3.cvssV3.baseSeverity",
        "cvssv3.cvssV3.confidentialityImpact",
        "cvssv3.cvssV3.integrityImpact",
        "cvssv3.cvssV3.privilegesRequired",
        "cvssv3.cvssV3.scope",
        "cvssv3.cvssV3.userInteraction",
        "published_at",
        "packages_count",
    ]

    @staticmethod
    def normalize_item(item):
        item_ = {key: val for key, val in item.items()}
        score = item_.get("cvssv3_score")
        if score:
            item_["score"] = score
            item_["score_type"] = "CVSS3"
        else:
            item_["score"] = item_.get("cvssv2_score")
            item_["score_type"] = "CVSS2"

        def fmt_package(pack):
            try:
                if "subdir" not in pack:
                    pack["subdir"] = ""
                return "{subdir}/{name}-{version} (sha258: {sha256})".format(**pack)
            except TypeError:
                return pack

        packages = [fmt_package(pack) for pack in item_.get("packages", [])]
        item_["packages"] = ", ".join(packages)
        return item_


class CVEFilesFormatter(FormatterBase):
    entity = "CVE Files"
    fmt_header_spacer = {
        "channel": "-" * 40,
        "artifact_family": "-" * 10,
        "cve_status": "-" * 10,
        "common_name": "-" * 18,
        "ckey": "-" * 32,
    }
    fmt_headers = (
        INITIAL_SPACE
        + "%(channel)-40s | %(artifact_family)-10s | %(cve_status)-10s | %(common_name)-18s | %(ckey)-32s"
    )

    keymap = {
        "cve_status": "Status",
        "artifact_family": "Family",
        "common_name": "Name",
        "ckey": "Path",
    }

    @staticmethod
    def normalize_item(item):
        item_ = {key: val for key, val in item.items()}
        item_["channel"] = (
            "%s/%s" % (item_["parent_channel_name"], item_["channel_name"])
            if item_["parent_channel_name"]
            else item_["channel_name"]
        )
        return item_


class PackageFilesFormatter(FormatterBase):
    entity = "Package Files"
    fmt_header_spacer = {
        "ckey": "-" * 60,
        "version": "-" * 10,
        "platform": "-" * 10,
    }
    fmt_headers = INITIAL_SPACE + "%(ckey)-60s | %(version)-10s | %(platform)-10s"

    keymap = {
        "ckey": "Path",
    }

    @staticmethod
    def normalize_item(item):
        return item


class PackageFilesFormatterWithCVE(FormatterBase):
    entity = "Package Files"
    fmt_header_spacer = {
        "ckey": "-" * 60,
        "cve_score": "-" * 9,
        "cve_status": "-" * 10,
        "version": "-" * 10,
        "platform": "-" * 10,
    }
    fmt_headers = (
        INITIAL_SPACE
        + "%(ckey)-60s | %(cve_score)-9s | %(cve_status)-10s | %(version)-10s | %(platform)-10s"
    )

    keymap = {"ckey": "Path", "cve_score": "CVE Score", "cve_status": "CVE Status"}

    @staticmethod
    def normalize_item(item):
        item_ = {key: val for key, val in item.items()}
        if item_["cve_status"] is None:
            item_["cve_status"] = "n/a"
        if item_["cve_score"] is None:
            item_["cve_score"] = "n/a"
        return item_


class HistoryFormatter(FormatterBase):
    entity = "Events"
    fmt_header_spacer = {
        "event_id": "-" * 36,
        "event_type": "-" * 20,
        "created": "-" * 32,
        "data_summary": "-" * 100,
    }
    fmt_headers = (
        INITIAL_SPACE
        + "%(event_id)-36s | %(event_type)20s | %(created)32s | %(data_summary)-50s"
    )

    keymap = {"event_id": "id", "event_type": "Type", "data_summary": "Summary"}

    fields = ["created", "data", "event_id", "event_type", "meta", "data_summary"]
    short_summary_keys = {"ckey", "artifact_family"}

    @classmethod
    def format_list_item(cls, item, short_summary=True):
        item_ = cls.normalize_item(item, short_summary)
        return cls.fmt_headers % item_

    @classmethod
    def normalize_item(cls, item, short_summary=True):
        item_ = {key: val for key, val in item.items()}
        if short_summary:
            item_["data_summary"] = "; ".join(
                [
                    "%s: %s" % (k, v)
                    for k, v in item["data"].items()
                    if k in cls.short_summary_keys
                ]
            )
        else:
            item_["data_summary"] = "; ".join(
                ["%s: %s" % (k, v) for k, v in item["data"].items()]
            )
        return item_

    @classmethod
    def format_list(cls, items, short_summary=True):
        lines = []
        lines.append(cls.format_list_headers())
        lines.append(cls.fmt_headers % cls.fmt_header_spacer)

        for item in items:
            lines.append(cls.format_list_item(item, short_summary))

        return "\n".join(lines)


class SettingsFormatter(FormatterBase):
    entity = "Admin Settings"
    fmt_header_spacer = {
        "key": "-" * 36,
        "value": "-" * 12,
    }
    fmt_headers = INITIAL_SPACE + "%(key)-36s | %(value)-12s"

    keymap = {"key": "Setting Name", "value": "Value"}
    fields = ["key", "value"]

    @classmethod
    def normalize_item(cls, item, short_summary=True):
        return item

    @classmethod
    def format_object_as_list(cls, obj):
        return cls.format_list([{"key": k, "value": v} for k, v in obj.items()])


def format_packages(packages, meta, logger):
    formatter = PackagesFormatter(logger)
    formatter.format(packages, meta)
    return formatter


def nested_get(item, path, default=None):
    """Returns the value of the nested dictionary, using path.

    Example:
        nested_get({"a":{"b": 42}}, "a.b") == 42
    """
    result = item
    for key in path.split("."):
        if key not in result:
            return default
        result = result[key]
    return result


def comma_string_to_list(s):
    """Returns a list of strings from the string, splitting by comma and removing trailing empty
        characters.

    Example:
        comma_string_to_list("flask, numpy ") == ["flask", "numpy"]
    """
    return [item.strip() for item in s.split(",") if item.strip()]


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
