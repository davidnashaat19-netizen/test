"""
Freelance Market Monitor — Deep Crawl Scraper
Platforms: Freelancer.com · Mostaqel.com
Output:    freelance_data.json

Usage:
    python scraper_ManualSubmission.py           # full crawl (up to 1000 pages)
    python scraper_ManualSubmission.py --test    # test mode (1 page per platform)
"""

# ---------------------------------------------------------------------------
# Imports & Logging
# ---------------------------------------------------------------------------
import json
import logging
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests

import nltk
for _pkg in ('stopwords', 'punkt', 'punkt_tab'):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords as _sw
    ARABIC_STOPWORDS = set(_sw.words('arabic'))
except Exception:
    ARABIC_STOPWORDS = set()

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------
@dataclass
class FreelanceProject:
    """
    Canonical record for one freelance project.

    DEEP CRAWLING NOTE: description_snippet is now replaced by
    full_description — the complete project description text extracted
    from the individual project detail page, not just a card summary.

    All fields that cannot be found are stored as None (null in JSON).
    """
    platform: str
    title: Optional[str] = None
    url: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    budget_currency: Optional[str] = None
    budget_type: Optional[str] = None
    skills: list = field(default_factory=list)
    category: Optional[str] = None
    posted_date: Optional[str] = None
    full_description: Optional[str] = None
    description_snippet: Optional[str] = None


# ---------------------------------------------------------------------------
# User-Agent Pool & create_driver
# ---------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]


def create_driver():
    """
    Headless Chrome driver.
    Selenium 4.6+ SeleniumManager auto-downloads the matching chromedriver.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,1024")
    chrome_options.add_argument("--force-renderer-accessibility")
    chrome_options.add_argument("--lang=ar,en")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    return driver


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def polite_sleep(min_s: float = 1.5, max_s: float = 4.0) -> None:
    duration = random.uniform(min_s, max_s)
    log.debug("  ↳ sleeping %.2f s …", duration)
    time.sleep(duration)


def find_element_by_selectors(parent, selectors: list):
    for selector in selectors:
        try:
            return parent.find_element(By.CSS_SELECTOR, selector)
        except NoSuchElementException:
            continue
    return None


def find_elements_by_selectors(parent, selectors: list):
    for selector in selectors:
        try:
            elements = parent.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                return elements
        except NoSuchElementException:
            continue
    return []


def get_text_by_selectors(parent, selectors: list) -> Optional[str]:
    el = find_element_by_selectors(parent, selectors)
    return el.text.strip() if el else None


def get_attribute_by_selectors(parent, selectors: list, attr: str) -> Optional[str]:
    el = find_element_by_selectors(parent, selectors)
    if el:
        val = el.get_attribute(attr)
        return val.strip() if val else None
    return None


def fetch_page_selenium(driver, url: str, retries: int = 3, backoff: float = 5.0) -> bool:
    for attempt in range(1, retries + 1):
        try:
            log.debug("Fetching listing with Selenium (attempt %d/%d): %s", attempt, retries, url)
            driver.get(url)
            time.sleep(random.uniform(2.5, 4.5))
            html = driver.page_source
            if html and len(html) > 200:
                return True
        except Exception as exc:
            log.warning("Attempt %d failed to fetch %s via Selenium: %s", attempt, url, exc)
        if attempt < retries:
            polite_sleep(backoff, backoff * 2)
    log.error("All %d Selenium fetch attempts failed for: %s", retries, url)
    return False


def fetch_detail_page_selenium(driver, url: str) -> bool:
    try:
        log.debug("Fetching detail page with Selenium: %s", url)
        driver.get(url)
        time.sleep(random.uniform(2.0, 3.5))
        html = driver.page_source
        if html and len(html) > 200:
            return True
    except Exception as exc:
        log.debug("Failed to fetch detail page %s: %s", url, exc)
    return False


# ---------------------------------------------------------------------------
# robots.txt Compliance
# ---------------------------------------------------------------------------

def _check_wildcard_disallow(robots_text: str, path: str) -> bool:
    in_wildcard_section = False
    for raw_line in robots_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("user-agent:"):
            agent = lower.split(":", 1)[1].strip()
            in_wildcard_section = (agent == "*")
        elif in_wildcard_section and lower.startswith("disallow:"):
            rule = line.split(":", 1)[1].strip()
            if "*" in rule:
                prefix = rule.split("*")[0]
                if path.startswith(prefix):
                    return True
    return False


def is_allowed_by_robots(base_url: str, path: str = "/") -> bool:
    robots_url = urljoin(base_url, "/robots.txt")
    target_url = urljoin(base_url, path)
    try:
        resp = requests.get(
            robots_url,
            timeout=10,
            headers={"User-Agent": random.choice(USER_AGENTS)},
        )
        if resp.status_code == 200:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(resp.text.splitlines())
            stdlib_allowed = rp.can_fetch("*", target_url)
            wildcard_blocked = _check_wildcard_disallow(resp.text, path)
            allowed = stdlib_allowed and not wildcard_blocked
            if not allowed:
                log.warning("robots.txt explicitly disallows: %s  (rule blocks %s)", target_url, path)
            else:
                log.info("robots.txt allows: %s", target_url)
            return allowed
        elif resp.status_code in (404, 410):
            log.info("robots.txt not found (HTTP %d) for %s → assuming allowed.", resp.status_code, base_url)
            return True
        elif resp.status_code in (401, 403):
            log.info("robots.txt returned HTTP %d for %s → treating as allowed.", resp.status_code, base_url)
            return True
        elif resp.status_code >= 500:
            log.warning("robots.txt server error HTTP %d for %s → failing open.", resp.status_code, base_url)
            return True
        else:
            log.warning("Unexpected HTTP %d fetching robots.txt for %s → allowing.", resp.status_code, base_url)
            return True
    except requests.exceptions.RequestException as exc:
        log.warning("Could not reach robots.txt at %s: %s → allowing.", robots_url, exc)
        return True


# ---------------------------------------------------------------------------
# Arabic Text Utilities
# ---------------------------------------------------------------------------

def normalize_arabic(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def clean_arabic_skills(raw_skills: list) -> list:
    seen = set()
    cleaned = []
    for s in raw_skills:
        s = normalize_arabic(s) or ""
        if not s or s.lower() in ARABIC_STOPWORDS:
            continue
        key = s.strip()
        if key not in seen:
            seen.add(key)
            cleaned.append(key)
    return cleaned


def extract_arabic_budget(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    tr = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(tr)


# ---------------------------------------------------------------------------
# Budget Parser
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Currency lookup table — ordered so longer / more-specific symbols are
# checked BEFORE their shorter prefixes:
#   "CA$" must precede "$" so Canadian dollars are not mis-tagged as USD.
#   "R$"  must precede bare "R" (ZAR) for the same reason.
#   Unicode symbols (₹ ₱ ₩ …) come first so they are never shadowed by an
#   ISO-code match lower in the list.
# ---------------------------------------------------------------------------
_CURRENCY_MAP = [
    # ── Unicode currency symbols ──────────────────────────────────────────
    ("₹",   "INR"),   # Indian Rupee
    ("₨",   "INR"),   # Rupee sign variant
    ("₱",   "PHP"),   # Philippine Peso
    ("₩",   "KRW"),   # Korean Won
    ("₺",   "TRY"),   # Turkish Lira
    ("₴",   "UAH"),   # Ukrainian Hryvnia
    ("₦",   "NGN"),   # Nigerian Naira
    # ── Compound $ symbols (must precede bare "$") ────────────────────────
    ("R$",  "BRL"),   # Brazilian Real
    ("CA$", "CAD"),   # Canadian Dollar
    ("A$",  "AUD"),   # Australian Dollar
    ("NZ$", "NZD"),   # New Zealand Dollar
    ("HK$", "HKD"),   # Hong Kong Dollar
    ("S$",  "SGD"),   # Singapore Dollar
    # ── Single-char / short symbols ───────────────────────────────────────
    ("$",   "USD"),   # US Dollar (after all $-prefixed variants)
    ("£",   "GBP"),   # British Pound
    ("€",   "EUR"),   # Euro
    ("RM",  "MYR"),   # Malaysian Ringgit
    ("د.إ", "AED"),   # UAE Dirham (Arabic)
    ("ر.س", "SAR"),   # Saudi Riyal (Arabic)
    ("ج.م", "EGP"),   # Egyptian Pound (Arabic)
    ("ريال","SAR"),   # Saudi Riyal (Arabic word)
    # ── ISO 4217 three-letter codes ───────────────────────────────────────
    ("INR", "INR"),   ("PKR", "PKR"),   ("BDT", "BDT"),
    ("CAD", "CAD"),   ("AUD", "AUD"),   ("NZD", "NZD"),
    ("HKD", "HKD"),   ("SGD", "SGD"),   ("MYR", "MYR"),
    ("AED", "AED"),   ("SAR", "SAR"),   ("EGP", "EGP"),
    ("NGN", "NGN"),   ("PHP", "PHP"),   ("KRW", "KRW"),
    ("BRL", "BRL"),   ("MXN", "MXN"),   ("ZAR", "ZAR"),
    ("CHF", "CHF"),   ("SEK", "SEK"),   ("NOK", "NOK"),
    ("DKK", "DKK"),   ("CZK", "CZK"),   ("PLN", "PLN"),
    ("TRY", "TRY"),   ("UAH", "UAH"),
    # ── Local shorthand ───────────────────────────────────────────────────
    ("SR",  "SAR"),   # Saudi Riyal shorthand
]


def clean_budget(raw: Optional[str]):
    if not raw:
        return None, None, None, "unknown"
    raw = extract_arabic_budget(raw)
    if not raw:
        return None, None, None, "unknown"
    raw = raw.strip()
    currency = None
    for symbol, code in _CURRENCY_MAP:
        if symbol in raw:
            currency = code
            break
    budget_type = "hourly" if re.search(r"/hr|/hour|per hour", raw, re.I) else "fixed"
    numbers = re.findall(r"[\d,]+\.?\d*", raw.replace(",", ""))
    nums = [float(n) for n in numbers if n]
    if len(nums) == 0:
        return None, None, currency, "unknown"
    elif len(nums) == 1:
        return nums[0], nums[0], currency, budget_type
    else:
        return min(nums), max(nums), currency, budget_type


# ---------------------------------------------------------------------------
# Scraper 1: Freelancer.com
# ---------------------------------------------------------------------------

FREELANCER_BASE   = "https://www.freelancer.com"
FREELANCER_SEARCH = "/jobs/"


def scrape_freelancer(driver, max_pages: int = 10, category_slug: str = "") -> list:
    projects = []
    search_path = FREELANCER_SEARCH + category_slug
    seen_urls: set = set()

    if not is_allowed_by_robots(FREELANCER_BASE, search_path):
        log.warning("Freelancer.com robots.txt blocks this path. Skipping.")
        return projects

    log.info("▶ Starting Freelancer.com DEEP CRAWL (max %d pages) …", max_pages)

    for page_num in range(1, max_pages + 1):
        page_url = f"{FREELANCER_BASE}{search_path}?page={page_num}"
        log.info("  [Listing] Page %d/%d → %s", page_num, max_pages, page_url)

        success = fetch_page_selenium(driver, page_url)
        if not success:
            log.warning("  Could not fetch listing page %d. Stopping.", page_num)
            break

        cards = find_elements_by_selectors(driver, [
            "div.JobSearchCard-item",
            "div[class*='job-card']",
            "li.job-wrap",
            "div.search-result-item"
        ])

        if not cards:
            log.warning("  No job cards on page %d. Site layout may have changed.", page_num)
            break

        log.info("  Found %d project cards on page %d.", len(cards), page_num)

        extracted_cards = []
        for card in cards:
            card_info = _parse_freelancer_card_selenium(card)
            if card_info:
                extracted_cards.append(card_info)

        for card_idx, card_info in enumerate(extracted_cards, start=1):
            project_url = card_info["url"]

            if not project_url:
                log.debug("    Card %d: no URL found, skipping detail fetch.", card_idx)
                bmin, bmax, currency, btype = clean_budget(card_info["raw_budget_card"])
                project = FreelanceProject(
                    platform="Freelancer.com",
                    title=card_info["title"],
                    url=None,
                    budget_min=bmin, budget_max=bmax,
                    budget_currency=currency, budget_type=btype,
                    skills=card_info["skills_card"],
                    category=card_info["category"],
                    posted_date=card_info["posted"],
                    full_description=None,
                    description_snippet=card_info["snippet"],
                )
                projects.append(project)
                continue

            if project_url in seen_urls:
                log.debug("    Card %d: duplicate URL skipped: %s", card_idx, project_url)
                continue
            seen_urls.add(project_url)

            log.debug("    [Deep Crawl] Card %d/%d — visiting detail page: %s",
                      card_idx, len(extracted_cards), project_url)
            polite_sleep(2, 5)

            detail_success = fetch_detail_page_selenium(driver, project_url)
            detail_data = _parse_freelancer_detail_selenium(driver) if detail_success \
                else {"full_description": None, "skills": [], "budget_raw": None}

            skills_final = detail_data["skills"] if detail_data["skills"] else card_info["skills_card"]
            raw_budget_final = detail_data["budget_raw"] or card_info["raw_budget_card"]
            bmin, bmax, currency, btype = clean_budget(raw_budget_final)

            project = FreelanceProject(
                platform="Freelancer.com",
                title=card_info["title"],
                url=project_url,
                budget_min=bmin, budget_max=bmax,
                budget_currency=currency, budget_type=btype,
                skills=skills_final,
                category=card_info["category"],
                posted_date=card_info["posted"],
                full_description=detail_data["full_description"],
                description_snippet=card_info["snippet"],
            )
            projects.append(project)
            log.debug("    ✔ Card %d — title: %s | skills: %d | desc_len: %d",
                      card_idx, (project.title or "")[:50], len(project.skills),
                      len(project.full_description or ""))

        log.info("  → %d projects collected so far.", len(projects))
        polite_sleep()

    log.info("✔ Freelancer.com DEEP CRAWL done. Total: %d projects.", len(projects))
    return projects


def _parse_freelancer_card_selenium(card) -> Optional[dict]:
    try:
        title = get_text_by_selectors(card, [
            "a.JobSearchCard-primary-heading-link",
            "h2.JobSearchCard-primary-heading a",
            "[class*='heading'] a"
        ])
        if not title:
            return None

        url = get_attribute_by_selectors(card, [
            "a.JobSearchCard-primary-heading-link",
            "h2.JobSearchCard-primary-heading a",
            "[class*='heading'] a",
            "a[href*='/projects/']"
        ], "href")

        raw_budget_card = get_text_by_selectors(card, [
            "div.JobSearchCard-primary-price",
            "[class*='price']",
            "[class*='budget']"
        ])

        skills_elements = find_elements_by_selectors(card, [
            "a.JobSearchCard-primary-tagsLink",
            "[class*='skill'] a",
            "[class*='tag'] a"
        ])
        skills_card = [el.text.strip() for el in skills_elements if el.text.strip()]

        category = get_text_by_selectors(card, [
            "a.JobSearchCard-primary-category",
            "[class*='category']"
        ])

        snippet = get_text_by_selectors(card, [
            "p.JobSearchCard-secondary-description",
            "[class*='description']"
        ])
        if snippet:
            snippet = snippet[:250]

        posted = get_text_by_selectors(card, ["span[class*='ago']", "time"])

        return {
            "title": title, "url": url, "raw_budget_card": raw_budget_card,
            "skills_card": skills_card, "category": category,
            "snippet": snippet, "posted": posted
        }
    except Exception as exc:
        log.warning("  Error parsing Freelancer card: %s", exc)
        return None


def _parse_freelancer_detail_selenium(driver) -> dict:
    result = {"full_description": None, "skills": [], "budget_raw": None}

    desc_element = find_element_by_selectors(driver, [
        "p.Project-description",
        "div.PageProjectViewLogout-projectDescription",
        "div.project-description",
        "[class*='ProjectDescription']",
        "[class*='project-description']",
        "div[class*='description'] p",
        "section.project-description"
    ])
    if desc_element:
        result["full_description"] = desc_element.text.strip()

    skills_elements = find_elements_by_selectors(driver, [
        "a[href*='/jobs/']",
        "a.skill-tag",
        "[class*='SkillTag']",
        "[class*='skill-tag']",
        "ul.skills-list li"
    ])
    result["skills"] = [el.text.strip() for el in skills_elements if el.text.strip()]

    budget_element = find_element_by_selectors(driver, [
        "h2.text-right",
        "h2.text-body-24",
        "[class*='PageProjectViewLogout-budget']",
        "[class*='project-budget']",
        "[class*='Budget']",
        "span[class*='price']"
    ])
    if budget_element:
        result["budget_raw"] = budget_element.text.strip()

    return result


# ---------------------------------------------------------------------------
# Scraper 2: Mostaqel.com
# ---------------------------------------------------------------------------

MOSTAQEL_BASE     = "https://mostaql.com"
MOSTAQEL_PROJECTS = "/projects"

_ARABIC_MONTHS = {
    'يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو',
    'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر',
}


def _looks_like_budget(text: str) -> bool:
    if not text or len(text) > 60:
        return False
    has_currency = bool(re.search(r'[$€£]', text)) or bool(
        re.search(r'\b(SAR|SR|ريال|ر\.س)\b', text)
    )
    if not has_currency:
        return False
    if not re.search(r'\d', text):
        return False
    if set(text.split()) & _ARABIC_MONTHS:
        return False
    if re.fullmatch(r'[\d,\.]+%', text.strip()):
        return False
    return True


def _find_budget_by_label(driver_or_card) -> Optional[str]:
    for xpath in [
        "//th[contains(., 'الميزانية')]/following-sibling::td[1]",
        "//dt[contains(., 'الميزانية')]/following-sibling::dd[1]",
        "//td[contains(., 'الميزانية')]/following-sibling::td[1]",
        "//li[contains(., 'الميزانية')]//span[last()]",
    ]:
        try:
            els = driver_or_card.find_elements(By.XPATH, xpath)
            for el in els:
                text = el.text.strip()
                if _looks_like_budget(text):
                    return text
        except Exception:
            continue

    for xpath in ["//*[contains(@class,'budget')]", "//*[contains(@class,'price')]"]:
        try:
            els = driver_or_card.find_elements(By.XPATH, xpath)
            for el in els:
                text = el.text.strip()
                if _looks_like_budget(text):
                    return text
        except Exception:
            continue

    for tag in ("td", "dd", "span"):
        try:
            for el in driver_or_card.find_elements(By.TAG_NAME, tag):
                text = el.text.strip()
                if _looks_like_budget(text):
                    return text
        except Exception:
            continue

    return None


def scrape_mostaqel(driver, max_pages: int = 10) -> list:
    projects = []
    seen_urls: set = set()

    if not is_allowed_by_robots(MOSTAQEL_BASE, MOSTAQEL_PROJECTS):
        log.warning("Mostaqel robots.txt blocks project listings. Skipping.")
        return projects

    log.info("▶ Starting Mostaqel.com DEEP CRAWL (max %d pages) …", max_pages)

    for page_num in range(1, max_pages + 1):
        page_url = f"{MOSTAQEL_BASE}{MOSTAQEL_PROJECTS}?page={page_num}"
        log.info("  [Listing] Page %d/%d → %s", page_num, max_pages, page_url)

        success = fetch_page_selenium(driver, page_url)
        if not success:
            log.warning("  Failed to fetch listing page %d. Stopping.", page_num)
            break

        cards = find_elements_by_selectors(driver, [
            "table.projects-table tbody tr",
            "div.project-row",
            "[class*='project-card']",
            "article.project"
        ])

        if not cards:
            log.warning("  No job cards found on page %d.", page_num)
            break

        log.info("  Found %d project cards on page %d.", len(cards), page_num)

        extracted_cards = []
        for card in cards:
            card_info = _parse_mostaqel_card_selenium(card)
            if card_info:
                extracted_cards.append(card_info)

        for card_idx, card_info in enumerate(extracted_cards, start=1):
            project_url = card_info["url"]

            if not project_url:
                log.debug("    Card %d: no URL found, skipping detail fetch.", card_idx)
                bmin, bmax, currency, btype = clean_budget(card_info["raw_budget_card"])
                project = FreelanceProject(
                    platform="Mostaqel.com",
                    title=card_info["title"],
                    url=None,
                    budget_min=bmin, budget_max=bmax,
                    budget_currency=currency, budget_type=btype,
                    skills=card_info["skills_card"],
                    category=card_info["category"],
                    posted_date=card_info["posted"],
                    full_description=None,
                    description_snippet=card_info["snippet"],
                )
                projects.append(project)
                continue

            if project_url in seen_urls:
                log.debug("    Card %d: duplicate URL skipped: %s", card_idx, project_url)
                continue
            seen_urls.add(project_url)

            log.debug("    [Deep Crawl] Card %d/%d — visiting detail page: %s",
                      card_idx, len(extracted_cards), project_url)
            polite_sleep(2, 5)

            detail_success = fetch_detail_page_selenium(driver, project_url)
            detail_data = _parse_mostaqel_detail_selenium(driver) if detail_success \
                else {"full_description": None, "skills": [], "budget_raw": None}

            skills_final = detail_data["skills"] if detail_data["skills"] else card_info["skills_card"]
            raw_budget_final = detail_data["budget_raw"] or card_info["raw_budget_card"]
            bmin, bmax, currency, btype = clean_budget(raw_budget_final)

            project = FreelanceProject(
                platform="Mostaqel.com",
                title=card_info["title"],
                url=project_url,
                budget_min=bmin, budget_max=bmax,
                budget_currency=currency, budget_type=btype,
                skills=skills_final,
                category=card_info["category"],
                posted_date=card_info["posted"],
                full_description=detail_data["full_description"],
                description_snippet=card_info["snippet"],
            )
            projects.append(project)
            log.debug("    ✔ Card %d — title: %s | skills: %d | desc_len: %d",
                      card_idx, (project.title or "")[:50], len(project.skills),
                      len(project.full_description or ""))

        log.info("  → %d projects collected so far.", len(projects))
        polite_sleep()

    log.info("✔ Mostaqel.com DEEP CRAWL done. Total: %d projects.", len(projects))
    return projects


def _parse_mostaqel_card_selenium(card) -> Optional[dict]:
    try:
        title = normalize_arabic(get_text_by_selectors(card, [
            "h2.project__title a", "h2 a", "a.project-title",
            "[class*='title'] a", "td.title-cell a"
        ]))
        if not title:
            return None

        url = get_attribute_by_selectors(card, [
            "h2.project__title a", "h2 a", "a.project-title",
            "[class*='title'] a", "a[href*='/projects/']"
        ], "href")

        raw_budget_card = extract_arabic_budget(_find_budget_by_label(card))

        skills_elements = find_elements_by_selectors(card, [
            "ul.project__skills li", "ul.skills-list li",
            "[class*='skill']", "span.tag", "a.tag"
        ])
        skills_card = clean_arabic_skills(
            [el.text.strip() for el in skills_elements if el.text.strip()]
        )

        category = normalize_arabic(get_text_by_selectors(card, [
            "a.project__category", "[class*='category'] a",
            "span.category", "td.category-cell a"
        ]))

        snippet = normalize_arabic(get_text_by_selectors(card, [
            "div.project__brief", "p.project-description",
            "[class*='description']", "div.carda__content p"
        ]))
        if snippet:
            snippet = snippet[:250]

        date_element = find_element_by_selectors(card, ["time", "[class*='date']"])
        posted = None
        if date_element:
            posted = date_element.get_attribute("datetime")
            if not posted:
                posted = date_element.text.strip()

        return {
            "title": title, "url": url, "raw_budget_card": raw_budget_card,
            "skills_card": skills_card, "category": category,
            "snippet": snippet, "posted": posted
        }
    except Exception as exc:
        log.warning("  Error parsing Mostaqel card: %s", exc)
        return None


def _parse_mostaqel_detail_selenium(driver) -> dict:
    result = {"full_description": None, "skills": [], "budget_raw": None}

    desc_element = find_element_by_selectors(driver, [
        "div.project__brief--full", "div.project-details__description",
        "[class*='project__description']", "[class*='ProjectDescription']",
        "div.carda__content p", "section.project-description",
        "[itemprop='description']"
    ])
    if desc_element:
        result["full_description"] = normalize_arabic(desc_element.text.strip())

    skills_elements = find_elements_by_selectors(driver, [
        "ul.project__skills li", "ul.skills-list li",
        "[class*='skill-tag']", "[class*='SkillsList'] li",
        "span.tag", "a.tag", "a[href*='/projects?skill=']"
    ])
    result["skills"] = clean_arabic_skills(
        [el.text.strip() for el in skills_elements if el.text.strip()]
    )

    raw_budget = _find_budget_by_label(driver)
    if raw_budget:
        result["budget_raw"] = extract_arabic_budget(raw_budget)

    return result


# ---------------------------------------------------------------------------
# JSON Exporter
# ---------------------------------------------------------------------------

def export_to_json(projects: list, filepath: str = "freelance_data.json") -> None:
    output = {
        "metadata": {
            "total_records": len(projects),
            "platforms": list({p.platform for p in projects}),
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": "2.0",
            "crawl_type": "deep",
        },
        "projects": [asdict(p) for p in projects],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("💾 Saved %d records → %s", len(projects), filepath)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("  Freelance Market Monitor — DEEP CRAWL Scraper Starting")
    log.info("  Crawl type  : Deep Crawling (following links)")
    log.info("  HTTP client : Selenium Chrome WebDriver (Headless)")
    log.info("  Parser      : Selenium (Native)")
    log.info("  Delay       : time.sleep()  [polite crawling]")
    log.info("  Note        : polite_sleep(2,5) between EVERY project visit")
    log.info("=" * 60)

    max_pages = 5
    if "--test" in sys.argv:
        max_pages = 1
        log.info("🧪 Running in TEST mode: limiting crawl to 1 page per platform.")

    driver = create_driver()
    all_projects = []
    freelancer_projects = []
    mostaqel_projects = []

    try:
        freelancer_projects = scrape_freelancer(driver, max_pages=max_pages)
        all_projects.extend(freelancer_projects)

        polite_sleep(3, 7)

        mostaqel_projects = scrape_mostaqel(driver, max_pages=max_pages)
        all_projects.extend(mostaqel_projects)

    finally:
        log.info("Closing Selenium WebDriver...")
        driver.quit()

    log.info("=" * 60)
    log.info("  DEEP CRAWL COMPLETE")
    log.info("  Freelancer.com : %d projects", len(freelancer_projects))
    log.info("  Mostaqel.com   : %d projects", len(mostaqel_projects))
    log.info("  TOTAL          : %d projects", len(all_projects))
    log.info("=" * 60)

    if not all_projects:
        log.warning("No data collected. The sites' HTML structure may have changed.")
        log.warning("Run with DEBUG logging: set level=logging.DEBUG above.")
        return

    export_to_json(all_projects, "freelance_data.json")


if __name__ == "__main__":
    main()
