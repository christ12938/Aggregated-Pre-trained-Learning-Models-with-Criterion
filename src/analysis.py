import pandas as pd
from tqdm import tqdm


result = {}
search_strings = ["lawsuit", "litigation", "plaintiff", "defendant", "case", "trial", "appeal", "court", "judge",
                  "jury", "attorney", "lawyer", "solicitor", "barrister", "counsel", "agreement", "contract", "clause",
                  "term", "condition", "obligation", "breach", "crime", "felony", "misdemeanor", "arrest", "bail",
                  "sentence", "probation", "divorce", "custody", "alimony", "marriage", "adoption", "guardianship",
                  "property", "estate", "land", "lease", "tenant", "landlord", "mortgage", "corporation", "partnership",
                  "bankruptcy", "merger", "acquisition", "liability", "patent", "copyright", "trademark",
                  "infringement", "license"]

search_strings_2 = ["constitution", "amendment", "bill of rights", "judicial review", "federalism",
                    "separation of powers", "due process",
                    "pollution", "emissions", "climate change", "endangered species", "waste disposal",
                    "sustainability", "natural resources",
                    "discrimination", "harassment", "minimum wage", "overtime", "unemployment", "workers' compensation",
                    "occupational safety",
                    "treaty", "diplomacy", "sanctions", "war crimes", "human rights", "international court", "genocide",
                    "income tax", "corporate tax", "tax evasion", "tax deduction", "tax credit", "tax return",
                    "sales tax",
                    "healthcare", "patient rights", "medical malpractice", "insurance", "public health",
                    "pharmaceuticals", "bioethics",
                    "freedom of speech", "privacy", "torture", "slavery", "racial discrimination", "gender equality",
                    "child rights"]

search_strings_3 = ["visa", "deportation", "naturalization", "citizenship", "asylum", "green card", "immigrant",
                    "will", "executor", "heir", "probate", "trust", "estate tax", "inheritance",
                    "discrimination", "equal protection", "civil liberties", "freedom of speech", "voting rights",
                    "desegregation", "affirmative action",
                    "shareholder", "dividend", "merger", "acquisition", "securities", "corporate governance",
                    "stock exchange",
                    "data protection", "privacy", "cybercrime", "hacking", "encryption", "internet regulation",
                    "cybersecurity",
                    "loan", "mortgage", "foreclosure", "bankruptcy", "securities", "investment", "interest rate",
                    "consumer rights", "warranty", "product liability", "fraud", "advertising", "debt collection",
                    "consumer protection",
                    "insurance policy", "claim", "premium", "coverage", "liability", "insurance fraud",
                    "risk assessment"]

search_strings_4 = ["negligence", "defamation", "liability", "damages", "personal injury", "malpractice",
                    "product liability"]

search_strings_5 = ['cardiovascular diseases',
                    'chronic kidney disease',
                    'chronic respiratory diseases',
                    'diabetes mellitus',
                    'digestive diseases',
                    'hiv/aids',
                    'hepatitis a/b/c/e',
                    'mental disorders',
                    'musculoskeletal disorders',
                    'neoplasms (cancer)',
                    'neurological disorders',
                    'hiv',
                    'disease']

search_strings_6 = ["annulment", "appellate", "arraignment", "bankruptcy", "codicil", "compensatory", "damages",
                    "discovery", "docket", "embezzlement", "espionage"]
search_strings_7 = ["abscond", "bailment", "conveyance", "decree", "estoppel", "fiduciary", "garnishment", "homestead",
                    "indemnity", "joinder", "kidnapping", "larceny", "misfeasance", "nolo", "ordinance", "plea",
                    "quorum", "replevin", "stipulation", "trustee", "unjust", "vagrancy", "writ", "xenodochium",
                    "yielding", "zealous"]
search_strings_8 = ["typhoid", "mumps", "pertussis", "zoster", "eczema", "allergy", "leprosy", "mumps", "influenza",
                    "malaria", "cholera", "dysentery", "asthma", "arthritis", "leukemia", "anemia", "obesity",
                    "alzheimer", "cancer", "stroke", "hepatitis", "parkinson", "schizophrenia", "fibrosis", "gastritis",
                    "ulcer", "measles", "diphtheria", "pneumonia", "bronchitis", "tuberculosis", "rabies", "polio",
                    "migraine", "osteoporosis", "hypertension", "diabetes", "hiv", "ebola", "zika", "dengue", "lyme",
                    "infection", "malnutrition", "ebola"]
final_string = ["nanotechnology", "pharmaceutics", "audiology", "viromics", "biophysics", "vaccinology", "cholera",
                "chikungunya", "dysentery", "cryptocurrency", "AI", "veganism", "freelancing", "hiking", "quilting",
                "upcycling", "e-sports", "motorcycling", "paranormal", "misdemeanor", "desegregation", "cybercrime",
                "replevin", "usury", "venire", "arraignment", "codicil", "vagrancy"]

fr_string = ["n'ouvrez", "nouvrez", "stickgoldet", "réveillerez", "reveillerez"]

fr_string_2 = ["histoire et révolution"]


def check_vocabs_contain_seeds(seeds: list, vocab_df: pd.DataFrame):
    matched_seeds = []
    for seed in tqdm(seeds, desc='Checking if vocab contains seeds'):
        for split_seed in seed.split():
            if vocab_df['vocab'].str.contains(split_seed).any():
                matched_seeds.append(seed)
                break
    print(matched_seeds)


def check_vocabs_has_seeds(seeds: list, vocab_df: pd.DataFrame):
    matched_seeds = []
    for seed in tqdm(seeds, desc='Checking if vocabs has seeds'):
        if seed in vocab_df['vocab'].values:
            matched_seeds.append(seed)
    print(matched_seeds)


if __name__ == '__main__':
    
    # Loop over each string in the list
    vocab_df = pd.read_pickle("/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/french_data/french_vocab.pkl")
    seeds = fr_string_2

    check_vocabs_contain_seeds(seeds=seeds, vocab_df=vocab_df)
    check_vocabs_has_seeds(seeds=seeds, vocab_df=vocab_df)
    
    print(vocab_df)
