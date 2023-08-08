import pandas as pd

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
# Loop over each string in the list
df = pd.read_pickle("/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_vocab.pkl")
for string in final_string:
    flag = True
    for split_string in string.split():
        contains_string = df['vocab'].str.contains(split_string).any()
        if contains_string:
            flag = False
            break
    if flag:
        result[string] = True

print(result)
