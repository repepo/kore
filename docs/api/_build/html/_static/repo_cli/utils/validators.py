import argparse
import datetime
from urllib.parse import urlparse

import requests


def check_cve_score(value):
    try:
        value = float(value)
        if value < 0 or value > 10:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("CVE Score has to be between 0.0 and 10.0")


def check_date_format(value):
    try:
        datetime.datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Date has to be in correct YYYY-MM-DD format")


def check_url(value):
    result = urlparse(value)
    if not result.netloc or not result.scheme:
        raise argparse.ArgumentTypeError("URL is not correct")
    return value


def check_url_exists(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except:
        return False


def check_proxy_url(value):
    if value == "":
        return value
    check_url(value)
    if not value.lower().startswith("http://"):
        raise argparse.ArgumentTypeError("URL must start with http://")
    return value
