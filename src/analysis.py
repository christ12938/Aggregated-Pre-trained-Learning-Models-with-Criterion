import pandas as pd
from tqdm import tqdm


scidocs_seeds = ['cardiovascular diseases',
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
                 'neurological disorders']


amazon_seeds = ['apps for android',
                'books',
                'cds and vinyl',
                'clothing, shoes and jewelry',
                'electronics',
                'health and personal care',
                'home and kitchen',
                'movies and tv',
                'sports and outdoors',
                'video games']


xlsum_fr_seeds = ["histoire et révolution", 
                  "faune et flore", 
                  "énergies renouvelables", 
                  "élections municipales",
                  "littérature contemporaine", 
                  "écoles élémentaires", 
                  "pâtisseries françaises", 
                  "châteaux historiques"]


def check_vocabs_contain_seeds(seeds: list, vocab_df: pd.DataFrame):
    matched_seeds = []
    for seed in tqdm(seeds, desc='Checking if vocab contains seeds'):
        for split_seed in seed.split():
            if vocab_df['vocab'].str.contains(split_seed).any():
                matched_seeds.append(seed)
                break
    print(matched_seeds)
    print()


def check_vocabs_has_seeds(seeds: list, vocab_df: pd.DataFrame):
    matched_seeds = []
    for seed in tqdm(seeds, desc='Checking if vocabs has seeds'):
        if seed in vocab_df['vocab'].values:
            matched_seeds.append(seed)
    print(matched_seeds)
    print()


if __name__ == '__main__':
    
    # print vocabs df
    vocab_df_1 = pd.read_pickle("/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_vocab.pkl")
    vocab_df_2 = pd.read_pickle("/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/amazon_vocab.pkl")
    vocab_df_3 = pd.read_pickle("/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/xlsum_fr_vocab.pkl")
    vocab_df_4 = pd.read_pickle("/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/merged_vocab.pkl")

    seeds_1 = scidocs_seeds
    seeds_2 = amazon_seeds
    seeds_3 = xlsum_fr_seeds

    print(vocab_df_1)
    print(vocab_df_2)
    print(vocab_df_3)
    print(vocab_df_4)
    
    check_vocabs_contain_seeds(seeds=seeds_1, vocab_df=vocab_df_1)
    check_vocabs_has_seeds(seeds=seeds_1, vocab_df=vocab_df_1)

    check_vocabs_contain_seeds(seeds=seeds_2, vocab_df=vocab_df_2)
    check_vocabs_has_seeds(seeds=seeds_2, vocab_df=vocab_df_2)       
    
    check_vocabs_contain_seeds(seeds=seeds_3, vocab_df=vocab_df_3)
    check_vocabs_has_seeds(seeds=seeds_3, vocab_df=vocab_df_3)       
