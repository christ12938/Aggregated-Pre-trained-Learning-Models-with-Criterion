from src.embeddings import SeedEmbeddings


def get_seeds():
    seeds = [
        'cardiovascular diseases',
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
        'disease'
    ]
    return seeds


if __name__ == "__main__":
    seed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=get_seeds(), model_options="bert_base")
    bert_base_seed_embeds.process_seeds()
    bert_base_seed_embeds.save_seed_embeddings(save_path=seed_bert_base_save_path)
    del bert_base_seed_embeds

    scibert_seed_embeds = SeedEmbeddings(seeds_list=get_seeds(), model_options="scibert")
    scibert_seed_embeds.process_seeds()
    scibert_seed_embeds.save_seed_embeddings(save_path=seed_scibert_save_path)
    del scibert_seed_embeds

