"""Text normalization, HTML extraction and spelling correction."""

import html
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem.snowball import RussianStemmer
from symspellpy import SymSpell, Verbosity

_NON_TEXT = re.compile(r"[^0-9a-zа-я]+")
_SPACES = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_ARTICLE_LINK = re.compile(r"/articles/(\d+)")
_STEMMER = RussianStemmer()
_NOISE = re.compile(
    r"\b(?:здравствуйте|здраствуйте|здравствуй|привет|"
    r"добрый\s+(?:день|вечер|утро)|подскажите|скажите|пожалуйста|"
    r"будьте\s+добры|можете\s+подсказать|хочу\s+узнать|"
    r"у\s+меня\s+вопрос)\b"
)


def normalize_text(value: object) -> str:
    """Lowercase text and keep only letters and digits."""
    text = html.unescape(str(value)).lower().replace("ё", "е")
    text = _URL.sub(" ", text)
    text = _NON_TEXT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def stem_text(value: object) -> str:
    """Apply Russian Snowball stemming to normalized text."""
    return " ".join(_STEMMER.stem(token) for token in str(value).split())


def remove_noise(value: object) -> str:
    """Remove greetings and common introductory phrases."""
    text = _NOISE.sub(" ", normalize_text(value))
    return _SPACES.sub(" ", text).strip()


class ArticleSpellCorrector:
    """Correct query typos using only the article vocabulary."""

    def __init__(self, articles: pd.DataFrame) -> None:
        """Build a correction dictionary from article words."""
        counts: Counter[str] = Counter()
        texts = articles["title_clean"].tolist() + articles["body_clean"].tolist()
        for text in texts:
            counts.update(token for token in text.split() if len(token) >= 4)

        self.words = set(counts)
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for word, count in counts.items():
            self.symspell.create_dictionary_entry(word, count)

    def fix(self, value: object) -> str:
        """Correct unknown words after removing conversational noise."""
        result: list[str] = []
        for token in remove_noise(value).split():
            if token in self.words or len(token) < 4 or token.isdigit():
                result.append(token)
                continue
            suggestions = self.symspell.lookup(
                token,
                Verbosity.TOP,
                max_edit_distance=2,
                include_unknown=True,
            )
            result.append(suggestions[0].term if suggestions else token)
        return " ".join(result)


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


def prepare_queries(
    frame: pd.DataFrame,
    corrector: ArticleSpellCorrector,
) -> pd.DataFrame:
    """Add original, noise-free and spell-corrected query views."""
    result = frame.copy()
    result["query_clean"] = result["query_text"].map(normalize_text)
    result["query_core"] = result["query_text"].map(remove_noise)
    result["query_spell"] = result["query_text"].map(corrector.fix)
    result["query_stem"] = result["query_clean"].map(stem_text)
    result["query_core_stem"] = result["query_core"].map(stem_text)
    result["query_spell_stem"] = result["query_spell"].map(stem_text)
    result["query_length"] = result["query_clean"].str.len()
    result["token_count"] = result["query_clean"].str.split().str.len()
    return result
