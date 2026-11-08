"""Synthetic data for track C: biographies (facts) and arithmetic / word problems (math).

Facts follow the bioS design of Allen-Zhu & Li ("Physics of Language Models 3.1"): each
person has six attributes; a biography states all six, one sentence each, with a random
template per sentence and the sentences in random order. Every template ends with the
attribute value, so a prompt is the template up to the value and recall is scored by
ranking all candidate values. Two templates per attribute are never used in training;
they measure recall under paraphrase.
"""
import random

FIRST = """Aaron Abigail Adrian Alma Amos Anika Arlo Astrid Basil Beatrix Bennett Bianca Caleb
Camille Cyrus Daphne Darius Delia Desmond Dorian Edith Elias Elsa Emeric Esme Ezra Felix Fiona
Gideon Greta Hollis Hazel Idris Imogen Ivo Jasper Juno Kai Kirsten Leander Leona Linus Lucia
Magnus Marlo Mira Nadia Nico Noemi Oren Orla Otto Pia Quentin Rhea Roman Rosalind Silas Sienna
Soren Tamsin Teodor Thea Ulric Uma Vance Vera Wendell Willa Xavier Yara Yusuf Zara Zeno Ines
Malcolm Odette""".split()
LAST = """Abernathy Ashby Bancroft Barlow Blackwood Calloway Carver Castellan Dalgren Delacroix
Ellison Everhart Fairbanks Fenwick Galloway Garrick Hadley Hargrove Holloway Ingram Jarvis
Kendrick Kingsley Lachance Langford Lockhart Marchetti Merriweather Montague Nakamura Northcott
Oakley Okonkwo Pemberton Prescott Quill Radcliffe Rasmussen Redfield Sandoval Sinclair
Stanhope Thackeray Thornbury Underhill Vandermeer Vasquez Wakefield Whitlock Winslow Yardley
Zielinski Achterberg Brannigan Cortazar Drummond Eastwood Falkner Grimaldi Halvorsen Iverson
Jablonski Koenig Lindqvist Moreau Novak Ortega Petrov Quintero Rourke Salazar Tremblay Ulloa
Valdez Whitaker Ximenez Youngblood Zamora""".split()
CITIES = """Amsterdam Ankara Athens Atlanta Auckland Baltimore Bangkok Barcelona Berlin Bogota
Boston Brisbane Brussels Budapest Cairo Calgary Chicago Copenhagen Dallas Denver Dublin Edinburgh
Florence Geneva Glasgow Hamburg Helsinki Houston Istanbul Jakarta Kyoto Lagos Lima Lisbon Lyon
Madrid Manila Marseille Melbourne Miami Milan Montreal Munich Nairobi Naples Oslo Ottawa Perth
Phoenix Prague Quebec Rome Santiago Seattle Seoul Stockholm Sydney Toronto Vienna Warsaw""".split()
UNIVERSITIES = [f"{p} {s}" for p, s in [
    ("Northfield", "University"), ("Westbrook", "College"), ("Ashford", "Institute of Technology"),
    ("Brightwater", "University"), ("Cedar Hill", "College"), ("Dunmore", "University"),
    ("Eastlake", "Polytechnic"), ("Fairview", "University"), ("Glenwood", "College"),
    ("Harborview", "University"), ("Ironwood", "Institute"), ("Juniper Valley", "University"),
    ("Kestrel", "College"), ("Lakeside", "University"), ("Maplewood", "Institute"),
    ("Northgate", "Polytechnic"), ("Oakridge", "University"), ("Pinecrest", "College"),
    ("Queensbury", "University"), ("Riverside", "Institute of Science"), ("Silverton", "University"),
    ("Thornfield", "College"), ("Upland", "University"), ("Valemont", "Institute"),
    ("Willowbrook", "University"), ("Yarrow", "College"), ("Alderidge", "University"),
    ("Blue Harbor", "Institute"), ("Coldspring", "University"), ("Deer Park", "College"),
    ("Elmstead", "University"), ("Foxhollow", "Institute"), ("Greystone", "University"),
    ("Highmoor", "College"), ("Ivybridge", "University"), ("Kingsbridge", "Polytechnic"),
    ("Larkspur", "University"), ("Moorfield", "College"), ("Newhaven", "Institute"),
    ("Redwood Coast", "University")]]
MAJORS = [m.strip() for m in """architecture|astronomy|biochemistry|biology|chemistry|civil engineering|
classics|computer science|economics|education|electrical engineering|English literature|
environmental science|film studies|finance|geography|geology|history|journalism|law|linguistics|
marine biology|mathematics|mechanical engineering|medicine|music|nursing|oceanography|pharmacy|
philosophy|physics|political science|psychology|public health|sociology|statistics|theatre|
urban planning|veterinary science|zoology""".replace("\n", "").split("|")]
EMPLOYERS = [f"{a} {b}" for a, b in [
    ("Apex", "Dynamics"), ("Bluefin", "Labs"), ("Cobalt", "Systems"), ("Driftwood", "Media"),
    ("Ember", "Energy"), ("Falcon", "Logistics"), ("Granite", "Bank"), ("Horizon", "Airlines"),
    ("Indigo", "Pharmaceuticals"), ("Jetstream", "Motors"), ("Keystone", "Insurance"),
    ("Lumen", "Optics"), ("Meridian", "Foods"), ("Nimbus", "Software"), ("Onyx", "Mining"),
    ("Pioneer", "Robotics"), ("Quasar", "Telecom"), ("Redstone", "Construction"),
    ("Sapphire", "Hotels"), ("Tidewater", "Shipping"), ("Umbra", "Studios"), ("Vertex", "Chemicals"),
    ("Wavecrest", "Health"), ("Zephyr", "Textiles"), ("Aurora", "Biotech"), ("Beacon", "Publishing"),
    ("Crescent", "Retail"), ("Delta", "Semiconductors"), ("Evergreen", "Farms"),
    ("Fjord", "Consulting"), ("Glacier", "Water"), ("Harbor", "Security"), ("Iris", "Cosmetics"),
    ("Juniper", "Networks"), ("Kodiak", "Outdoor"), ("Lighthouse", "Analytics"),
    ("Monarch", "Aerospace"), ("Northstar", "Railways"), ("Orchid", "Fashion"),
    ("Polaris", "Instruments"), ("Quarry", "Cement"), ("Riverbend", "Paper"),
    ("Summit", "Capital"), ("Timber", "Furniture"), ("Unity", "Hospitals"), ("Vanguard", "Toys"),
    ("Willow", "Gardens"), ("Yellowstone", "Tours"), ("Zenith", "Electronics"), ("Atlas", "Freight")]]
YEARS = [str(y) for y in range(1900, 2000)]

ATTRS = {
    "birth_year": YEARS, "birth_city": CITIES, "university": UNIVERSITIES,
    "major": MAJORS, "employer": EMPLOYERS, "work_city": CITIES,
}
TRAIN_T = {
    "birth_year": ["{n} was born in the year {v}.", "{n} came into the world in {v}.",
                   "The year {n} was born was {v}.", "{n}'s year of birth is {v}."],
    "birth_city": ["{n} was born in {v}.", "{n} spent their early years in {v}.",
                   "{n}'s hometown is {v}.", "{n} grew up in the city of {v}."],
    "university": ["{n} studied at {v}.", "{n} graduated from {v}.",
                   "{n} received their degree from {v}.", "{n} attended {v}."],
    "major": ["{n} majored in {v}.", "{n} chose to study {v}.",
              "{n}'s field of study was {v}.", "{n} earned a degree in {v}."],
    "employer": ["{n} works for {v}.", "{n} is employed by {v}.",
                 "{n} has a job at {v}.", "{n} earns a living working at {v}."],
    "work_city": ["{n} works in the city of {v}.", "{n}'s office is in {v}.",
                  "{n} commutes to work in {v}.", "Every morning {n} goes to work in {v}."],
}
HELDOUT_T = {
    "birth_year": ["Records show that {n} was born in {v}.", "Ask {n} when they were born and they will say {v}."],
    "birth_city": ["The birthplace of {n} is {v}.", "{n} is originally from {v}."],
    "university": ["{n} is an alumnus of {v}.", "The university {n} went to is {v}."],
    "major": ["The subject {n} specialized in is {v}.", "At university, {n} focused on {v}."],
    "employer": ["The company that pays {n} is {v}.", "{n} is a member of the staff at {v}."],
    "work_city": ["The workplace of {n} is located in {v}.", "{n} has a job based in {v}."],
}


def people(n, seed=0):
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    return [{"name": nm, **{a: rng.choice(vals) for a, vals in ATTRS.items()}} for nm in names[:n]]


def sentence(person, attr, template):
    return template.format(n=person["name"], v=person[attr])


def bio(person, rng):
    attrs = list(ATTRS)
    rng.shuffle(attrs)
    return " ".join(sentence(person, a, rng.choice(TRAIN_T[a])) for a in attrs)


def fact_prompts(person, heldout=True):
    """(attr, prompt, value) for every attribute and template of one kind."""
    out = []
    for a, ts in (HELDOUT_T if heldout else TRAIN_T).items():
        for t in ts:
            s = sentence(person, a, t)
            v = person[a]
            out.append((a, s[: s.rindex(v)].rstrip(), v))
    return out


# ---- math --------------------------------------------------------------------------
KIDS = """Lily Tom Ben Mia Sam Anna Max Lucy Jack Emma Leo Zoe Tim Sara Ella Finn Ruby Noah""".split()
ITEMS = """apples toys books stickers cookies marbles pencils balloons shells flowers cards eggs""".split()


def num(rng, digits):
    return rng.randint(10 ** (digits - 1) if digits > 1 else 0, 10 ** digits - 1)


def arith_line(rng, op=None, da=None, db=None):
    op = op or rng.choice("+-*")
    if op == "*":
        da, db = da or rng.randint(1, 3), db or rng.randint(1, 3)
    else:
        da, db = da or rng.randint(1, 5), db or rng.randint(1, 5)
    a, b = num(rng, da), num(rng, db)
    c = a + b if op == "+" else a - b if op == "-" else a * b
    return f"{a}{op}{b}", str(c)


def word_problem(rng):
    k, k2 = rng.sample(KIDS, 2)
    it = rng.choice(ITEMS)
    kind = rng.randrange(4)
    if kind == 0:
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        return (f"{k} has {a} {it}. {k} gets {b} more {it}. How many {it} does {k} have now?",
                f"{a}+{b}={a + b}. The answer is {a + b}.")
    if kind == 1:
        a = rng.randint(2, 999)
        b = rng.randint(1, a)
        return (f"{k} had {a} {it} and gave {b} to {k2}. How many {it} does {k} have left?",
                f"{a}-{b}={a - b}. The answer is {a - b}.")
    if kind == 2:
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        return (f"There are {a} boxes with {b} {it} in each box. How many {it} are there in total?",
                f"{a}*{b}={a * b}. The answer is {a * b}.")
    a, b, d = rng.randint(2, 30), rng.randint(2, 30), rng.randint(1, 99)
    x = a * b
    return (f"{k} buys {a} packs of {it} with {b} in each pack, and then finds {d} more. "
            f"How many {it} does {k} have?", f"{a}*{b}={x}. {x}+{d}={x + d}. The answer is {x + d}.")


def math_doc(rng):
    lines = []
    for _ in range(rng.randint(4, 10)):
        if rng.random() < 0.6:
            q, a = arith_line(rng)
            lines.append(f"{q}={a}")
        else:
            q, a = word_problem(rng)
            lines.append(f"{q} {a}")
    return "\n".join(lines)
