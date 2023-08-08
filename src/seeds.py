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


def get_merged_list_seeds():
    # Merge the lists
    merged_list = list(set(get_scidocs_seeds() + get_amazon_seeds()))
    # Shuffle the merged list
    #random.shuffle(merged_list)
    print(merged_list)
    return merged_list


if __name__ == "__main__":
    scidocs_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scidocs_seed_tensors_bert_base_uncased.pkl"
    scidocs_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scidocs_seed_tensors_scibert_uncased.pkl"
    scidocs_seed_embed_legalbert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scidocs_seed_tensors_legalbert_uncased.pkl"

    amazon_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/amazon_seed_tensors_bert_base_uncased.pkl"
    amazon_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/amazon_seed_tensors_scibert_uncased.pkl"
    amazon_seed_embed_legalbert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/amazon_seed_tensors_legalbert_uncased.pkl"

    twitter_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/twitter_seed_tensors_bert_base_uncased_alt_seeds.pkl"
    twitter_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/twitter_seed_tensors_scibert_uncased_alt_seeds.pkl"
    twitter_seed_embed_legalbert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/twitter_seed_tensors_legalbert_uncased.pkl"

    legal_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/legal_seed_tensors_bert_base_uncased.pkl"
    legal_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/legal_seed_tensors_scibert_uncased.pkl"
    legal_seed_embed_legalbert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/legal_seed_tensors_legalbert_uncased.pkl"

    merged_seed_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_bert_base_uncased.pkl"
    merged_seed_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_scibert_uncased.pkl"
    merged_seed_embed_legalbert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_legalbert_uncased.pkl"

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="scibert")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_scibert_save_path)
    del bert_base_seed_embeds

    """
    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="scibert")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_scibert_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_twitter_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=twitter_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_twitter_seeds(), model_options="scibert")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=twitter_seed_embed_scibert_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="scibert")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_scibert_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_legal_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=legal_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_bert_base_save_path)
    del bert_base_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_twitter_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=twitter_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_legal_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=legal_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_scibert_save_path)
    del scibert_seed_embeds

    legalbert_seed_embeds = SeedEmbeddings(seeds_list=get_twitter_seeds(), model_options="legalbert")
    legalbert_seed_embeds.process_seeds()
    legalbert_seed_embeds.save_seed_embeddings(save_path=twitter_seed_embed_legalbert_save_path)
    del legalbert_seed_embeds

    legalbert_seed_embeds = SeedEmbeddings(seeds_list=get_scidocs_seeds(), model_options="legalbert")
    legalbert_seed_embeds.process_seeds()
    legalbert_seed_embeds.save_seed_embeddings(save_path=scidocs_seed_embed_legalbert_save_path)
    del legalbert_seed_embeds

    legalbert_seed_embeds = SeedEmbeddings(seeds_list=get_legal_seeds(), model_options="legalbert")
    legalbert_seed_embeds.process_seeds()
    legalbert_seed_embeds.save_seed_embeddings(save_path=legal_seed_embed_legalbert_save_path)
    del legalbert_seed_embeds

    legalbert_seed_embeds = SeedEmbeddings(seeds_list=get_merged_list_seeds(), model_options="legalbert")
    legalbert_seed_embeds.process_seeds()
    legalbert_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_legalbert_save_path)
    del legalbert_seed_embeds
    """
