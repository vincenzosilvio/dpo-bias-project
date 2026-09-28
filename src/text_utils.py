"""
Shared text analysis for building pairs and reviewing data.

Pronoun counting is only a valid label when the occupation-holder is the
ONLY person the pronouns can refer to (README, lesson #9). So on top of the
pronoun filters, `analyze()` identifies the character's first name and
rejects texts with more than one named person.

Names are detected with the Kantrowitz names corpus (resources/names, see
its README for license/credit) restricted to proper nouns via spaCy POS
tags. spaCy NER alone was tested and rejected: it tagged "Marcus" as a
place and missed "Priya".
"""

import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMES_DIR = REPO_ROOT / "resources" / "names"

MALE_PRONOUNS = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)

# Safety net for real, well-known people (observed: Mark Zuckerberg for
# "CEO"). Not exhaustive -- extend with names found in manual review.
REAL_PERSON_BLOCKLIST = [
    "mark zuckerberg", "elon musk", "bill gates", "jeff bezos",
    "tim cook", "sundar pichai", "satya nadella", "sam altman",
    "larry page", "sergey brin", "jack ma", "warren buffett",
    "steve jobs", "richard branson", "indra nooyi", "mary barra",
    "ginni rometty", "sheryl sandberg",
    "florence nightingale", "marie curie", "albert einstein",
    "amelia earhart", "neil armstrong",
    # added after the first full short-form review (pair 79: Michael Jackson
    # in a firefighter text). Famous people the model may reach for.
    "michael jackson", "elvis presley", "taylor swift", "barack obama",
    "donald trump", "joe biden", "oprah winfrey", "michael jordan",
    "lebron james", "serena williams", "tom hanks", "tom cruise",
    "brad pitt", "leonardo dicaprio", "beyonce", "madonna", "lady gaga",
    "kim kardashian", "princess diana", "queen elizabeth", "john lennon",
    "paul mccartney", "freddie mercury", "martin luther king", "abraham lincoln",
    "george washington", "isaac newton", "stephen hawking", "nikola tesla",
    "thomas edison", "ada lovelace", "grace hopper", "alan turing",
    "linus torvalds", "gordon ramsay", "jamie oliver",
]
# Whole-word match: a plain substring test matched "jack ma" inside
# "Mechanic Jack made" (second short-form run).
REAL_PERSON_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in REAL_PERSON_BLOCKLIST) + r")\b", re.IGNORECASE)

REFUSAL_PATTERN = re.compile(
    r"\b(as an ai|ai language model|i'm sorry|i am sorry|i cannot|i can't "
    r"(help|assist|write)|i'm unable|i am unable)\b",
    re.IGNORECASE,
)

# Dialogue is stripped before checking for first-person narration.
QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
FIRST_PERSON = re.compile(r"\bI\b|\b(?i:my|me|myself)\b")

# Other-person nouns: flagged in review, NOT excluded (a nurse's text may
# mention "patients" harmlessly). Whether they cause misattribution in the
# short set is measured by the manual audit, and tightened if needed.
OTHER_PERSON = re.compile(
    r"\b(colleague|coworker|co-worker|boss|manager|assistant|friend|patient|"
    r"customer|client|visitor|guest|student|child|kid|son|daughter|wife|"
    r"husband|partner|mother|father|mom|dad|sister|brother|passenger|"
    r"man|woman|boy|girl|stranger)s?\b",
    re.IGNORECASE,
)


def pronoun_counts(text):
    return {
        "male": len(MALE_PRONOUNS.findall(text)),
        "female": len(FEMALE_PRONOUNS.findall(text)),
    }


def pronoun_gender(text):
    """'male' / 'female' (pure), 'mixed', or 'none'."""
    c = pronoun_counts(text)
    if c["male"] and c["female"]:
        return "mixed"
    if c["male"]:
        return "male"
    if c["female"]:
        return "female"
    return "none"


def mentions_real_person(text):
    return bool(REAL_PERSON_PATTERN.search(text))


def has_first_person_narration(text):
    return bool(FIRST_PERSON.search(QUOTED.sub(" ", text)))


def exclusion_reason(record, min_pronoun_count):
    """Text-level reasons a completion can't be used at all, or None."""
    text = record["completion"]
    if not text.strip():
        return "empty"
    if record.get("truncated", False):
        return "truncated"
    if mentions_real_person(text):
        return "real_person"
    if REFUSAL_PATTERN.search(text):
        return "refusal"
    if has_first_person_narration(text):
        return "first_person"
    c = pronoun_counts(text)
    if c["male"] + c["female"] < min_pronoun_count:
        return "no_signal"
    if c["male"] and c["female"]:
        return "mixed"
    return None


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def name_lists():
    """(male_only, female_only, unisex) sets of first names."""
    male = set(open(NAMES_DIR / "male.txt").read().split())
    female = set(open(NAMES_DIR / "female.txt").read().split())
    return frozenset(male - female), frozenset(female - male), frozenset(male & female)


def name_gender(name):
    """'male', 'female', 'unisex' or None (not in the corpus)."""
    male, female, unisex = name_lists()
    if name in male:
        return "male"
    if name in female:
        return "female"
    if name in unisex:
        return "unisex"
    return None


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


TITLES = {"Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Sir", "Madam", "Mx"}


# Corpus names that are also everyday words: accepted as names only when
# spaCy NER tags them PERSON. The PROPN tag is not enough: sentence-initial
# "Hope filled the room" is tagged PROPN (observed).
AMBIGUOUS_NAMES = {
    "will", "may", "hope", "grace", "joy", "faith", "rose", "dawn", "summer",
    "page", "bill", "mark", "art", "sue", "pat", "jack", "guy", "earl",
    "june", "april", "august", "ray", "dean", "sky", "sunny", "autumn",
    "winter", "holly", "ivy", "iris", "lily", "violet", "hazel", "ruby",
    "amber", "crystal", "pearl", "ginger", "honey", "angel", "chase",
    "hunter", "miles", "norm", "rich", "gay", "constance", "prudence",
    "patience", "liberty", "destiny", "harmony", "melody", "story", "lee",
    "ash", "reed", "day", "love", "star", "brook", "river", "stone", "cliff",
    "don", "frank", "grant", "gene", "cole", "wade", "drew", "rob", "sterling",
    "happy", "lucky", "merry", "bliss", "precious", "charity", "honor",
    "justice", "royal", "sage", "blessing", "promise", "trinity",
}
TITLE_GENDER = {"Mr": "male", "Mrs": "female", "Ms": "female", "Miss": "female"}

# Common surnames that the names corpus also lists as first names ("Smith",
# "Patel" are male there). After a title they are surnames, and they never
# serve as replacement first names (second short-form run: "Sarah" ->
# "Smith" in 16 pairs). First-name-heavy surnames (James, Thomas, Scott...)
# are deliberately left out.
COMMON_SURNAMES = frozenset("""
Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez
Hernandez Lopez Gonzalez Wilson Anderson Taylor Moore Jackson Perez Thompson
White Harris Sanchez Clark Ramirez Lewis Robinson Walker Wright Torres Nguyen
Hill Flores Green Adams Nelson Baker Hall Rivera Campbell Mitchell Carter
Roberts Patel Chen Kim Wang Li Zhang Liu Singh Kumar Khan Rogers Cooper Reed
Bailey Bell Murphy Parker Evans Edwards Collins Stewart Morris Cook Morgan
Peterson Gray Ramos Watson Brooks Sanders Price Bennett Wood Barnes Ross
Henderson Coleman Jenkins Perry Powell Long Patterson Hughes Washington
Butler Simmons Foster Gonzales Bryant Alexander Russell Griffin Diaz Hayes
Myers Ford Hamilton Graham Sullivan Wallace Woods Cole West Jordan Owens
Reynolds Fisher Ellis Harrison Gibson McDonald Cruz Marshall Ortiz Gomez
Murray Freeman Wells Webb Simpson Stevens Tucker Porter Hunter Hicks Crawford
Henry Boyd Mason Warren Dixon Burns Gordon Shaw Holmes Rice Robertson Hunt
Black Daniels Palmer Mills Nichols Grant Knight Ferguson Stone Hawkins
Dunn Perkins Hudson Spencer Gardner Stephens Payne Pierce Berry Matthews
Arnold Wagner Willis Ray Watkins Olson Carroll Duncan Snyder Hart
Cunningham Bradley Lane Andrews Ruiz Harper Fox Riley Armstrong Carpenter
Weaver Greene Lawrence Elliott Chavez Sims Austin Peters Kelley Franklin
Lawson Fields Ryan Schmidt Carr Vasquez Castillo Wheeler Chapman Oliver
Montgomery Richards Williamson Johnston Banks Meyer Bishop McCoy Howell
Morrison Hansen Fernandez Garza Harvey Little Burton Stanley Nguyen George
Jacobs Reid Fuller Lynch Dean Gilbert Garrett Romero Welch Larson Frazier
Burke Hanson Day Mendoza Moreno Bowman Medina Fowler Brewer Hoffman Carlson
Silva Pearson Holland Douglas Fleming Jensen Vargas Byrd Davidson Hopkins
May Terry Herrera Wade Soto Walters Curtis Neal Caldwell Lowe Jennings
Barnett Graves Jimenez Horton Shelton Barrett Obrien Castro Sutton Gregory
McKinney Lucas Miles Craig Rodriquez Chambers Holt Lambert Fletcher Watts
Bates Hale Rhodes Pena Beck Newman Haynes McDaniel Mendez Bush Vaughn Parks
Dawson Santiago Norris Hardy Love Steele Curry Powers Schultz Barker Guzman
Page Munoz Ball Keller Chandler Weber Leonard Walsh Lyons Ramsey Wolfe
Schneider Mullins Benson Sharp Bowen Daniel Barber Cummings Hines Baldwin
Griffith Valdez Hubbard Salazar Reeves Warner Stevenson Burgess Santos Tate
Cross Garner Mann Mack Moss Thornton Dennis McGee Farmer Delgado Aguilar
Vega Glover Manning Cohen Harmon Rodgers Robbins Newton Todd Blair Higgins
Ingram Reese Cannon Strickland Townsend Potter Goodwin Walton Rowe Hampton
Ortega Patton Swanson Joseph Francis Goodman Maldonado Yates Becker Erickson
Hodges Rios Conner Adkins Webster Norman Malone Hammond Flowers Cobb Moody
Quinn Blake Maxwell Pope Floyd Osborne Paul McCarthy Guerrero Lindsey Estrada
Sandoval Gibbs Tyler Gross Fitzgerald Stokes Doyle Sherman Saunders Wise
Colon Gill Alvarado Greer Padilla Simon Waters Nunez Ballard Schwartz McBride
Houston Christensen Klein Pratt Briggs Parsons McLaughlin Zimmerman French
Buchanan Moran Copeland Roy Pittman Brady McCormick Holloway Brock Poole
Frank Logan Owen Bass Marsh Drake Wong Jefferson Park Morton Abbott Sparks
Patrick Norton Huff Clayton Massey Lloyd Figueroa Carson Bowers Roberson
Barton Tran Lamb Harrington Casey Boone Cortez Clarke Mathis Singleton
Wilkins Cain Bryan Underwood Hogan McKenzie Collier Luna Phelps McGuire Allison
Bridges Wilkerson Nash Summers Atkins Wilcox Pitts Conley Marquez Burnett
Richard Cochran Chase Davenport Hood Gates Clay Ayala Sawyer Roman Vazquez
Dickerson Hodge Acosta Flynn Espinoza Nicholson Monroe Wolf Morrow Kirk
Randall Anthony Whitaker Oconnor Skinner Ware Molina Kirby Huffman Bradford
Charles Gilmore Dominguez Oneal Bruce Lang Combs Kramer Heath Hancock
Gallagher Gaines Shaffer Short Wiggins Mathews McClain Fischer Wall Small
Melton Hensley Bond Dyer Cameron Grimes Contreras Christian Wyatt Baxter
Snow Mosley Shepherd Larsen Hoover Beasley Glenn Petersen Whitehead Meyers
Keith Garrison Vincent Shields Horn Savage Olsen Schroeder Hartman Woodard
Mueller Kemp Deleon Booth Patel Calhoun Wiley Eaton Cline Navarro Harrell
Lester Humphrey Parrish Duran Hutchinson Hess Dorsey Bullock Robles Beard
Dalton Avila Vance Rich Blackwell York Johns Blankenship Trevino Salinas
Campos Pruitt Moses Callahan Golden Montoya Hardin Guerra McDowell Carey
Stafford Gallegos Henson Wilkinson Booker Merritt Miranda Atkinson Orr Decker
Hobbs Preston Tanner Knox Pacheco Stephenson Glass Rojas Serrano Marks
Hickman English Sweeney Strong Prince McClure Conway Walter Roth Maynard
Farrell Lowery Hurst Nixon Weiss Trujillo Ellison Sloan Juarez Winters
McLean Randolph Leon Boyer Villarreal McCall Gentry Carrillo Kent Ayers Lara
Shannon Sexton Pace Hull Leblanc Browning Velasquez Leach Chang House Sellers
Herring Noble Foley Bartlett Mercado Landry Durham Walls Barr McKee Bauer
Rivers Everett Bradshaw Pugh Velez Rush Estes Dodson Morse Sheppard Weeks
Camacho Bean Barron Livingston Middleton Spears Branch Blevins Chen Kerr
McConnell Hatfield Harding Ashley Solis Herman Frost Giles Blackburn William
Pennington Woodward Finley McIntosh Koch Best Solomon McCullough Dudley Nolan
Blanchard Rivas Brennan Mejia Kane Benton Joyce Buckley Haley Valentine
Maddox Russo McKnight Buck Moon McMillan Crosby Berg Dotson Mays Roach Church
Chan Richmond Meadows Faulkner Oneill Knapp Kline Barry Ochoa Jacobson Gay
Avery Hendricks Horne Shepard Hebert Cherry Cardenas McIntyre Whitney Waller
Holman Donaldson Cantu Terrell Morin Gillespie Fuentes Tillman Sanford
Bentley Peck Key Salas Rollins Gamble Dickson Battle Santana Cabrera Cervantes
Howe Hinton Hurley Spence Zamora Yang McNeil Suarez Case Petty Gould McFarland
Sampson Carver Bray Rosario Macdonald Stout Hester Melendez Dillon Farley
Hopper Galloway Potts Bernard Joyner Stein Aguirre Osborn Mercer Bender
Franco Rowland Sykes Benjamin Travis Pickett Crane Sears Mayo Dunlap Hayden
Wilder McKay Coffey McCarty Ewing Cooley Vaughan Bonner Cotton Holder Stark
Ferrell Cantrell Fulton Lynn Lott Calderon Rosa Pollard Hooper Burch Mullen
Fry Riddle Levy David Duke Odonnell Guy Michael Britt Frederick Daugherty
Berger Dillard Alston Jarvis Frye Riggs Chaney Odom Duffy Fitzpatrick
Valenzuela Merrill Mayer Alford McPherson Acevedo Donovan Barrera Albert Cote
Reilly Compton Raymond Mooney McGowan Craft Cleveland Clemons Wynn Nielsen
Baird Stanton Snider Rosales Bright Witt Stuart Hays Holden Rutledge Kinney
Clements Castaneda Slater Hahn Emerson Conrad Burks Delaney Pate Lancaster
Sweet Justice Tyson Sharpe Whitfield Talley Macias Irwin Burris Ratliff
McCray Madden Kaufman Beach Goff Cash Bolton McFadden Levine Good Byers
Kirkland Kidd Workman Carney Dale McLeod Holcomb England Finch Head Burt
Hendrix Sosa Haney Franks Sargent Nieves Downs Rasmussen Bird Hewitt Lindsay
Le Foreman Valencia Oneil Delacruz Vinson Dejesus Hyde Forbes Gilliam Guthrie
Wooten Huber Barlow Boyle McMahon Buckner Rocha Puckett Langley Knowles Cooke
Velazquez Whitley Noel Vang Nakamura Tanaka Yamamoto Sato Suzuki Watanabe
Ito Kobayashi Singh Sharma Gupta Mehta Shah Rao Reddy Iyer Das Bose Ahmed
Hassan Ali Hussain Rahman Ibrahim Mohamed Rossi Russo Ferrari Esposito
Bianchi Romano Colombo Ricci Marino Greco Bruno Gallo Conti Costa Mancini
Muller Schmidt Schneider Fischer Weber Wagner Becker Hoffmann Dubois Moreau
Laurent Lefebvre Petit Durand Leroy Ivanov Petrov Novak Kowalski Nowak
Andersson Johansson Nilsson Hansen Jensen Larsen Kowalczyk Oconnell Kelly
Murphy Walsh Byrne Ryan Doherty Kennedy Lynch Quinn Obrien Harper Sinclair
Blackwood Ashford Whitmore Hawthorne Sterling Wellington Pemberton
""".split()) - frozenset("""
Albert Alexander Allison Anthony Ashley Austin Avery Barry Benjamin Blair Blake
Bradley Brady Bruce Bryan Cameron Carey Carroll Carson Casey Chandler Charles Chase
Christian Clay Cole Conrad Craig Curtis Dale Daniel David Dean Delaney Dennis Dillon
Donovan Douglas Drake Dudley Duncan Elliott Ellis Emerson Everett Finley Floyd Francis
Frank Franklin Frederick George Gilbert Glenn Gordon Grant Gregory Guy Haley Harrison
Harvey Hayden Henry Herman Hunter Irwin Jefferson Jordan Joseph Joyce Keith Kelly
Kelley Kent Kim Kirby Kirk Lane Lara Lawrence Leon Leonard Leroy Lester Lindsay Lindsey
Lloyd Logan Lucas Luna Lynn Marshall Mason Maxwell May Miles Miranda Michael Mitchell
Monroe Morgan Morris Moses Neal Nelson Noel Nolan Norman Oliver Owen Patrick Paul Perry
Pierce Preston Quinn Randall Ray Raymond Reese Reid Rich Richard Riley Rosa Rosario Roy
Russell Ryan Sawyer Shannon Sherman Simon Solomon Spencer Stanley Sterling Stuart Tanner
Tate Taylor Terry Todd Travis Tucker Tyler Tyson Valentine Vance Vaughn Vincent Wade
Wallace Walter Warren Whitney Wiley William Wyatt Wynn Valencia Heath Gay Love Cherry
Berry Hester Gates Rowe Bell Gray Greer Lamb Page Brooks Britt Gill Merrill Noble
""".split())   # also common first names: "Miss Ashley" is a first name


def surname_after_title(token_text, title_gender):
    """Is a single word after a gendered title a surname ("Mrs. Smith")
    rather than a first name ("Miss Alice")?"""
    if token_text in COMMON_SURNAMES:
        return True
    return name_gender(token_text) not in (title_gender, "unisex")
NON_NAME_POS = {"VERB", "AUX", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT", "NUM"}


def _is_name_part(tok):
    """
    Part of a person reference. PROPN alone is not enough: spaCy tags
    sentence-initial "Emily" as an adverb (observed), so a capitalized
    corpus name counts unless its POS rules it out or it is an everyday word.
    """
    if tok.lower_ in AMBIGUOUS_NAMES:
        return tok.ent_type_ == "PERSON"
    if tok.pos_ == "PROPN":
        return True
    if not tok.text[:1].isupper() or name_gender(tok.text) is None:
        return False
    return tok.pos_ not in NON_NAME_POS


def _quoted_token_ids(doc):
    """Token indices inside quotes: a slogan like "Always Happy" is not a name."""
    ids, inside = set(), False
    for tok in doc:
        if tok.text in ('"', "“", "”"):
            inside = not inside if tok.text == '"' else tok.text == "“"
            continue
        if inside:
            ids.add(tok.i)
    return ids


def _propn_spans(doc):
    """Maximal runs of consecutive name parts ("Dr. Emily Chen"), outside quotes."""
    quoted = _quoted_token_ids(doc)
    spans, cur = [], []
    for tok in doc:
        # A quoted nickname inside a name ('Elisabeth "Betty" Rogers') must not
        # split it into two people: skip it without closing the current span.
        if cur and (tok.i in quoted or tok.text in ('"', "“", "”")):
            continue
        if tok.i not in quoted and _is_name_part(tok):
            cur.append(tok)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def find_names(doc):
    """
    First names of the people in a spaCy doc, as {name: gender}.

    Each run of consecutive proper nouns is one person reference: its first
    token found in the names corpus (skipping titles and words like
    "Nurse") is the first name; later tokens are surnames and ignored --
    otherwise "Emily Chen" would count as two people ("Chen" is in the
    corpus). A run with no corpus name but a gendered title ("Mr. Johnson")
    yields the surname with the title's gender (lesson #10). Otherwise a run
    that spaCy tags PERSON yields an unknown name (gender None).
    """
    person_idx = {t.i for ent in doc.ents if ent.label_ == "PERSON" for t in ent}
    found = {}
    for span in _propn_spans(doc):
        tokens = [t for t in span if t.text.rstrip(".") not in TITLES]
        name = next((t.text for t in tokens if name_gender(t.text) is not None), None)
        title = next((TITLE_GENDER[t.text.rstrip(".")] for t in span
                      if t.text.rstrip(".") in TITLE_GENDER), None)
        if title is not None and len(tokens) == 1 and surname_after_title(tokens[0].text, title):
            # Gendered title + surname ("Mrs. Smith", "Ms. Patel"): the title
            # decides, even when the corpus lists the word as a male first
            # name. "Miss Alice" is a first name and takes the branch below.
            found[tokens[0].text] = title
        elif name is not None:
            found[name] = name_gender(name)
        elif tokens and title is not None:
            found[tokens[-1].text] = title
        elif tokens and any(t.i in person_idx for t in tokens):
            found.setdefault(tokens[0].text, None)
    return found


def titled_surname(doc, name):
    """True if `name` only occurs as 'Title Surname' ("Mrs. Smith")."""
    for span in _propn_spans(doc):
        tokens = [t for t in span if t.text.rstrip(".") not in TITLES]
        title = next((TITLE_GENDER[t.text.rstrip(".")] for t in span
                      if t.text.rstrip(".") in TITLE_GENDER), None)
        if [t.text for t in tokens] == [name] and title and surname_after_title(name, title):
            return True
    return False


def name_spans(doc, name):
    """Non-title tokens of each person span that contains `name`."""
    out = []
    for span in _propn_spans(doc):
        tokens = [t for t in span if t.text.rstrip(".") not in TITLES]
        if any(t.text == name for t in tokens):
            out.append(tokens)
    return out


def analyze(record, min_pronoun_count, nlp=None):
    """
    Full per-completion analysis. Returns a dict:
        reason  -- exclusion reason or None
        gender  -- pure pronoun gender ('male'/'female') when usable
        name, name_gender -- the single character's first name, if usable
        other_person_flag -- other-person noun present (review only)
    Order matters: cheap text filters first, spaCy only for survivors.
    """
    out = {"reason": None, "gender": None, "name": None, "name_gender": None,
           "other_person_flag": False}
    reason = exclusion_reason(record, min_pronoun_count)
    if reason:
        out["reason"] = reason
        return out
    text = record["completion"]
    out["gender"] = pronoun_gender(text)
    out["other_person_flag"] = bool(OTHER_PERSON.search(text))

    doc = (nlp or get_nlp())(text)
    names = find_names(doc)
    if len(names) == 0:
        out["reason"] = "no_name"
    elif len(names) > 1:
        out["reason"] = "multi_person"
    else:
        (name, g), = names.items()
        out["name"], out["name_gender"] = name, g
        if g is None:
            out["reason"] = "unknown_name"
        elif g != "unisex" and g != out["gender"]:
            # "Sarah ... he": name and pronouns disagree -> either a second
            # person or incoherent text. Either way the label is unreliable.
            out["reason"] = "name_pronoun_mismatch"
    return out
