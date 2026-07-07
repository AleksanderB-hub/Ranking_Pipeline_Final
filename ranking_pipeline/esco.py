"""Streamlined ESCO data manager.

Downloads (once, into a per-user cache directory) and loads the ESCO skills
classification, exposing only the skill preferred labels -- that is all the
pipeline needs for its candidate vocabulary, and loading just one column
keeps memory use and start-up time low.
"""

import logging
import urllib.parse
import zipfile
from enum import Enum
from pathlib import Path

import appdirs
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Language enum."""

    EN = "en"
    DE = "de"
    FR = "fr"
    IT = "it"
    ES = "es"
    BG = "bg"
    CS = "cs"
    DA = "da"
    ET = "et"
    EL = "el"
    GA = "ga"
    HR = "hr"
    LV = "lv"
    LT = "lt"
    HU = "hu"
    MT = "mt"
    NL = "nl"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    SK = "sk"
    SL = "sl"
    FI = "fi"
    SV = "sv"
    IS = "is"
    NO = "no"
    AR = "ar"
    UK = "uk"
    JA = "ja"
    KO = "ko"
    ZH = "zh"


class ESCO:
    """ESCO data manager with flexible data location and caching.

    Streamlined to only load and provide skill preferred labels.
    """

    BASE_URL = "https://ec.europa.eu/esco/download/"
    SUPPORTED_ESCO_LANGUAGES: tuple[Language, ...] = (
        Language.BG, Language.ES, Language.CS, Language.DA, Language.DE,
        Language.ET, Language.EL, Language.EN, Language.FR, Language.GA,
        Language.HR, Language.IT, Language.LV, Language.LT, Language.HU,
        Language.MT, Language.NL, Language.PL, Language.PT, Language.RO,
        Language.SK, Language.SL, Language.FI, Language.SV, Language.IS,
        Language.NO, Language.AR, Language.UK,
    )

    def __init__(
        self,
        version: str = "1.2.0",
        language: Language = Language.EN,
        data_dir: str | None = None,
        auto_download: bool = True,
    ):
        if version == "default":
            version = "1.2.0"

        self.version = version
        self.auto_download = auto_download

        self.language = language
        assert self.language in self.SUPPORTED_ESCO_LANGUAGES, (
            f"Language {self.language.value} not supported by ESCO. "
            f"Supported languages: {[lang.value for lang in self.SUPPORTED_ESCO_LANGUAGES]}"
        )

        if data_dir:
            self.base_path = Path(data_dir)
        else:
            cache_dir = appdirs.user_cache_dir("wteb")
            self.base_path = Path(cache_dir) / "esco"

        self.data_path = self.base_path / version / self.language

        self._initialize_data()

    def _initialize_data(self):
        """Initialize ESCO data, downloading if necessary."""
        if self._load_data_from_path(self.data_path):
            return

        if not self.auto_download:
            raise RuntimeError(f"ESCO data not found at {self.data_path} and auto_download=False")

        try:
            logger.debug(f"ESCO v{self.version} ({self.language.value}) not found in cache.")
            logger.debug(f"Downloading to {self.data_path}...")
            self._download_esco_data()

            if self._load_data_from_path(self.data_path):
                logger.debug("Successfully downloaded and loaded ESCO data.")
                return
        except Exception as e:
            logger.debug(f"Failed to download ESCO data: {e}")
            raise RuntimeError(
                f"Could not load or download ESCO data for version {self.version}, "
                f"language {self.language.value}. Try setting auto_download=True or "
                f"manually downloading data to {self.data_path}"
            ) from e

    def _load_data_from_path(self, path: Path) -> bool:
        """Try to load ESCO skills data from the given path. Returns True if successful."""
        skills_path = path / f"skills_{self.language.value}.csv"

        if not skills_path.exists():
            logger.debug(f"ESCO skills data not found at {skills_path}")
            return False

        # Load only the necessary columns to save memory.
        self.skills_df = pd.read_csv(
            skills_path,
            usecols=["conceptUri", "preferredLabel"]
        )

        # Remove NaNs in case of dirty data.
        self.skills_df = self.skills_df[self.skills_df["preferredLabel"].notna()]

        # URI -> preferred label mapping.
        self.uri_to_skill = dict(zip(self.skills_df["conceptUri"], self.skills_df["preferredLabel"]))

        logger.debug(f"Loaded ESCO v{self.version} ({self.language.value}) from {path}")
        return True

    def _create_download_url(self) -> str:
        """Create the download URL for the specific version and language."""
        filename = f"ESCO dataset - v{self.version} - classification - {self.language.value} - csv.zip"
        encoded_filename = urllib.parse.quote(filename)
        return f"{self.BASE_URL}{encoded_filename}"

    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download a file from URL to output_path. Returns True if successful."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(output_path, "wb") as f,
                tqdm(
                    desc=f"Downloading ESCO v{self.version} ({self.language.value})",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during download: {e}")
            return False

    def _unzip_file(self, zip_path: Path) -> bool:
        """Extract zip file to the appropriate dataset directory."""
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Extract everything: zipfile can't easily pull out just the
                # skills CSV without knowing exact internal paths.
                zip_ref.extractall(self.data_path)

            return True

        except zipfile.BadZipFile as e:
            logger.warning(f"Invalid zip file: {e}")
            return False
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            return False

    def _download_esco_data(self) -> None:
        """Download ESCO data for the specified version and language."""
        download_cache = self.base_path / "downloads"
        download_cache.mkdir(parents=True, exist_ok=True)

        filename = f"ESCO_dataset_v{self.version}_classification_{self.language.value}_csv.zip"
        zip_path = download_cache / filename

        if not zip_path.exists():
            url = self._create_download_url()
            if not self._download_file(url, zip_path):
                raise RuntimeError(f"Failed to download ESCO data from {url}")

        if not self._unzip_file(zip_path):
            raise RuntimeError(f"Failed to extract ESCO data from {zip_path}")

    def get_skills_vocabulary(self) -> list[str]:
        """Get list of all skill preferred labels in deterministic sorted order."""
        return sorted(self.skills_df["preferredLabel"].unique().tolist())
