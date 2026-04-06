"""
Constants for OASIS Agent Profile Generator
"""

# MBTI types list
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

# Common countries list
COUNTRIES = [
    "China", "US", "UK", "Japan", "Germany", "France",
    "Canada", "Australia", "Brazil", "India", "South Korea"
]

# Individual entity types (require generating specific personas)
INDIVIDUAL_ENTITY_TYPES = [
    "student", "alumni", "professor", "person", "publicfigure",
    "expert", "faculty", "official", "journalist", "activist",
    "politician", "scientist", "researcher", "athlete", "artist",
    "musician", "author", "entrepreneur", "investor", "diplomat",
    "celebrity", "ceo", "executive", "regulator",
]

# Keywords in entity type names that indicate an individual
INDIVIDUAL_TYPE_KEYWORDS = [
    "founder", "forecaster", "user", "trader", "influencer",
    "analyst", "advisor", "leader", "critic", "advocate",
    "commentator", "blogger", "developer", "engineer",
]

# Group/institutional entity types (require generating representative account personas)
GROUP_ENTITY_TYPES = [
    "university", "governmentagency", "organization", "ngo",
    "mediaoutlet", "company", "institution", "group", "community",
    "agency", "platform", "network", "protocol", "framework",
    "fund", "exchange", "consortium", "coalition",
]
