import random


def get_scidocs_seeds():
    seeds = [
        'cardiovascular diseases',
        'chronic kidney disease',
        'chronic respiratory diseases',
        'diabetes mellitus',
        'digestive diseases',
        'hiv/aids',
        'sexually transmitted diseases',
        'hepatitis a/b/c/d/e',
        'mental disorders',
        'musculoskeletal disorders',
        'neoplasms (cancer)',
        'neurological disorders',
    ]
    return seeds


def get_amazon_seeds():
    seeds = ['apps for android',
             'books and readings',
             'cds and vinyl',
             'clothing, shoes and jewelry',
             'electronic devices',
             'health and personal care',
             'home and kitchen',
             'movies and tv',
             'sports and outdoors',
             'video games']
    return seeds


def get_french_seeds():
    seeds = ["histoire et révolution", "faune et flore", "énergies renouvelables", "élections municipales",
             "littérature contemporaine", "écoles élémentaires", "pâtisseries françaises", "châteaux historiques"]
    topics_fr = [
        #"conflits armés",
#        "conflit israélo-arabe",
        "israël et arabe",
        "énergies renouvelables",
        "justice pénale",
        "élections municipales",
        "voiture autonome",
        "progrès médicaux",
        "innovation technologique",
        ]

    return topics_fr
