SECTOR_CODES = {
    "001": (0, "전기전자", 300),
    "002": (1, "금융업", 100),
    "003": (2, "서비스업", 400),
    "004": (3, "의약품", 150),
    "005": (4, "운수창고", 50),
    "006": (5, "유통업", 80),
    "007": (6, "건설업", 70),
    "008": (7, "철강금속", 60),
    "009": (8, "기계", 100),
    "010": (9, "화학", 150),
    "011": (10, "섬유의복", 40),
    "012": (11, "음식료품", 60),
    "013": (12, "비금속광물", 30),
    "014": (13, "종이목재", 20),
    "015": (14, "운수장비", 80),
    "016": (15, "통신업", 20),
    "017": (16, "전기가스업", 15),
    "018": (17, "제조업(기타)", 200),
    "019": (18, "농업임업어업", 10),
    "020": (19, "광업", 10),
}

NUM_SECTORS = 20


def get_sector_id(sector_code: str) -> int:
    if sector_code in SECTOR_CODES:
        return SECTOR_CODES[sector_code][0]
    return -1


def get_sector_name(sector_id: int) -> str:
    for code, (sid, name, _) in SECTOR_CODES.items():
        if sid == sector_id:
            return name
    return "Unknown"
