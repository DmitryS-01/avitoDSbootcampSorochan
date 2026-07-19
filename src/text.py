"""Text normalization and HTML field extraction."""

import html
import re
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem.snowball import RussianStemmer

_NON_TEXT = re.compile(r"[^0-9a-zа-я]+")
_SPACES = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_ARTICLE_LINK = re.compile(r"/articles/(\d+)")
_STEMMER = RussianStemmer()


def normalize_text(value: object) -> str:
    """Lowercase text and keep only letters and digits."""
    text = html.unescape(str(value)).lower().replace("ё", "е")
    text = _URL.sub(" ", text)
    text = _NON_TEXT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def stem_text(value: object) -> str:
    """Apply Russian Snowball stemming to normalized text."""
    return " ".join(_STEMMER.stem(token) for token in str(value).split())


@dataclass(frozen=True)
class HtmlFields:
    """Useful text fields extracted from an article body."""

    visible: str
    important: str
    anchors: str
    alt_text: str
    intro: str
    internal_links: str


def extract_html_fields(value: object) -> HtmlFields:
    """Extract visible text, headings, links and article references."""
    soup = BeautifulSoup(str(value), "html.parser")
    for tag in soup(["script", "style", "input"]):
        tag.decompose()

    visible = normalize_text(soup.get_text(" "))
    important = normalize_text(
        " ".join(
            tag.get_text(" ")
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong", "label"])
        )
    )
    anchors = normalize_text(" ".join(tag.get_text(" ") for tag in soup.find_all("a")))
    alt_text = normalize_text(
        " ".join(tag.get("alt", "") for tag in soup.find_all("img"))
    )
    links = sorted({int(item) for item in _ARTICLE_LINK.findall(str(value))})
    return HtmlFields(
        visible=visible,
        important=important,
        anchors=anchors,
        alt_text=alt_text,
        intro=visible[:1_200],
        internal_links=" ".join(map(str, links)),
    )


def prepare_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Create the processed article table used by all models."""
    fields = articles["body"].map(extract_html_fields)
    result = articles[["article_id"]].copy()
    result["title_clean"] = articles["title"].map(normalize_text)
    result["body_clean"] = fields.map(lambda item: item.visible)
    result["important_clean"] = fields.map(lambda item: item.important)
    result["anchor_clean"] = fields.map(lambda item: item.anchors)
    result["alt_clean"] = fields.map(lambda item: item.alt_text)
    result["intro_clean"] = fields.map(lambda item: item.intro)
    result["internal_links"] = fields.map(lambda item: item.internal_links)
    result["title_stem"] = result["title_clean"].map(stem_text)
    result["body_stem"] = result["body_clean"].map(stem_text)
    result["body_length"] = result["body_clean"].str.len()
    return result


def prepare_queries(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized and stemmed query columns."""
    result = frame.copy()
    result["query_clean"] = result["query_text"].map(normalize_text)
    result["query_stem"] = result["query_clean"].map(stem_text)
    result["query_length"] = result["query_clean"].str.len()
    result["token_count"] = result["query_clean"].str.split().str.len()
    return result
