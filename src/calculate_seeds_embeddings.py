from embeddings import SeedEmbeddings
from seeds import get_scidocs_seeds, get_amazon_seeds, get_french_seeds


if __name__ == "__main__":

    scidocs_seed_embed_bert_base_save_path = "embeddings/scidocs_seed_tensors_bert_large_uncased.pkl"
    scidocs_seed_embed_scibert_save_path = "embeddings/scidocs_seed_tensors_scibert_uncased.pkl"
    scidocs_seed_embed_flaubert_save_path = "embeddings/scidocs_seed_tensors_flaubert_large_uncased.pkl"

    amazon_seed_embed_bert_base_save_path = "embeddings/amazon_seed_tensors_bert_large_uncased.pkl"
    amazon_seed_embed_scibert_save_path = "embeddings/amazon_seed_tensors_scibert_uncased.pkl"
    amazon_seed_embed_flaubert_save_path = "embeddings/amazon_seed_tensors_flaubert_large_uncased.pkl"

    french_seed_embed_bert_base_save_path = "embeddings/french_seed_tensors_bert_large_uncased.pkl"
    french_seed_embed_scibert_save_path = "embeddings/french_seed_tensors_scibert_uncased.pkl"
    french_seed_embed_flaubert_save_path = "embeddings/french_seed_tensors_flaubert_large_uncased.pkl"

    merged_seed_embed_bert_base_save_path = "embeddings/merged_seed_tensors_bert_large_uncased.pkl"
    merged_seed_embed_scibert_save_path = "embeddings/merged_seed_tensors_scibert_uncased.pkl"
    merged_seed_embed_flaubert_save_path = "embeddings/merged_seed_tensors_flaubert_large_uncased.pkl"

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

    bert_base_seed_embeds = SeedEmbeddings(seeds_list=merged_seeds, model_options="bert_base")
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

    scibert_seed_embeds = SeedEmbeddings(seeds_list=merged_seeds, model_options="scibert")
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

    flaubert_seed_embeds = SeedEmbeddings(seeds_list=merged_seeds, model_options="flaubert")
    flaubert_seed_embeds.process_seeds()
    flaubert_seed_embeds.save_seed_embeddings(save_path=merged_seed_embed_flaubert_save_path)
    del flaubert_seed_embeds

