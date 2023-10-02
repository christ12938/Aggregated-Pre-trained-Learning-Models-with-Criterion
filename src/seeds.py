import random

from src.embeddings import SeedEmbeddings


def get_scidocs_seeds():

    seeds = [
        'cardiovascular diseases',
        'chronic kidney disease',
        'chronic respiratory diseases',
        'diabetes mellitus',
        'digestive diseases',
        'hiv/aids',
        'sexually transmitted diseases',
        'hepatitis',
        'mental disorders',
        'musculoskeletal disorders',
        'neoplasms',
        'neurological disorders',
    ]

    #seeds = ["nanotechnology", "pharmaceutics", "audiology", "viromics", "biophysics", "vaccinology", "cholera",
    #         "chikungunya", "dysentery"]
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
    return seeds


def get_merged_seeds(seeds: list):
    # Merge the lists
    merged_list = list(set(seeds))
    # Shuffle the merged list
    #random.shuffle(merged_list)
    print(merged_list)
    return merged_list


if __name__ == "__main__":

    scidocs_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scidocs_seed_tensors_bert_base_uncased.pkl"
    scidocs_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scidocs_seed_tensors_scibert_uncased.pkl"
    scidocs_seed_embed_flaubert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scidocs_seed_tensors_flaubert_uncased.pkl"

    amazon_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/amazon_seed_tensors_bert_base_uncased.pkl"
    amazon_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/amazon_seed_tensors_scibert_uncased.pkl"
    amazon_seed_embed_flaubert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/amazon_seed_tensors_flaubert_uncased.pkl"

    french_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/french_seed_tensors_bert_base_uncased.pkl"
    french_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/french_seed_tensors_scibert_uncased.pkl"
    french_seed_embed_flaubert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/french_seed_tensors_flaubert_uncased.pkl"

    merged_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_bert_base_uncased.pkl"
    merged_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_scibert_uncased.pkl"
    merged_seed_embed_flaubert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_flaubert_uncased.pkl"
    
    merged_seeds = get_scidocs_seeds() + get_amazon_seeds() + get_french_seeds()


    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_amazon_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=amazon_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_french_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=french_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_merged_seeds(seeds=merged_seeds), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_amazon_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=amazon_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_french_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=french_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_merged_seeds(seeds=merged_seeds), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    flaubert_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="flaubert")
    flaubert_seed_embeds.process_seeds()
    flaubert_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_flaubert_save_path)
    del flaubert_seed_embeds

    flaubert_seed_embeds = SeedEmbeddings(seeds_list=get_amazon_seeds(), model_options="flaubert")
    flaubert_seed_embeds.process_seeds()
    flaubert_seed_embeds.save_seed_embeddings(save_path=amazon_seed_embed_flaubert_save_path)
    del flaubert_seed_embeds

    flaubert_seed_embeds = SeedEmbeddings(seeds_list=get_french_seeds(), model_options="flaubert")
    flaubert_seed_embeds.process_seeds()
    flaubert_seed_embeds.save_seed_embeddings(save_path=french_seed_embed_flaubert_save_path)
    del flaubert_seed_embeds

    flaubert_seed_embeds = SeedEmbeddings(seeds_list=get_merged_seeds(seeds=merged_seeds), model_options="flaubert")
    flaubert_seed_embeds.process_seeds()
    flaubert_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_flaubert_save_path)
    del flaubert_seed_embeds
