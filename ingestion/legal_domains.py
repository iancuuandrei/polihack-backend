from __future__ import annotations


LEGAL_DOMAIN_REGISTRY: dict[str, str] = {
    "ro.codul_muncii": "munca",
    "ro.constitutia": "constitutional",
    "ro.codul_civil": "civil",
    "ro.codul_fiscal": "fiscal",
    "ro.og_2_2001": "contraventional",
    "ro.oug_195_2002": "rutier",
    "ro.lege_190_2018": "protectia_datelor",
}


def get_registered_legal_domain(law_id: str | None) -> str | None:
    if not law_id:
        return None
    return LEGAL_DOMAIN_REGISTRY.get(law_id)
