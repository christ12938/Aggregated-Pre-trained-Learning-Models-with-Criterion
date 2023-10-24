import random


def get_scidocs_seeds():
    """
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
    """
    seeds = ["nanotechnology", "pharmaceutics", "audiology", "viromics", "biophysics", "vaccinology", "cholera",
             "chikungunya", "dysentery"]
    return seeds


def get_amazon_seeds():
    seeds = ['apps for android',
             'books',
             'cds and vinyl',
             'clothing, shoes and jewelry',
             'electronics',
             'health and personal care',
             'home and kitchen',
             'movies and tv',
             'sports and outdoors',
             'video games']
    return seeds


def get_twitter_seeds():
    """
    seeds = ['food',
             'shop and service',
             'travel and transport',
             'college and university',
             'nightlife spot',
             'residence',
             'outdoors and recreation',
             'arts and entertainment',
             'professional and other places']
             """
    seeds = ["cryptocurrency", "veganism", "freelancing", "hiking", "quilting", "upcycling", "e-sports",
             "motorcycling", "paranormal"]
    return seeds


def get_legal_seeds():
    """
    seeds = ['federalism', 'bioethics', 'misdemeanor', 'alimony', 'desegregation', 'cybercrime', 'doping',
             'malpractice']
             """
    seeds = ["misdemeanor", "desegregation", "cybercrime", "replevin", "usury", "venire", "arraignment", "codicil",
             "vagrancy", 'federalism']
    return seeds


def get_french_seeds():
    seeds = ["histoire et révolution", "faune et flore", "énergies renouvelables", "élections municipales",
             "littérature contemporaine", "écoles élémentaires", "pâtisseries françaises", "châteaux historiques"]
    topics_fr = [
        "changement climatique",
        "éducation nationale",
        "justice pénale",
        "technologies émergentes",
        "affaires étrangères",
        "conflits armés",
        "littérature contemporaine",
        "arts visuels",
        "jeux vidéo",
        "voyages spatiaux"
                ]

    return seeds
