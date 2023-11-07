#TODO: Fix packages naming and directories
from scidocs_preproc import SciDocsPreprocess
from amazon_preproc import AmazonPreprocess
from french_news_preproc import FrenchNewsPreprocess
from merged_preproc import MergedPreprocess
from seetopic_scidocs_preproc import SeetopicScidocsPreprocess


if __name__ == '__main__':
    
    # preprocess scidocs data
    scidocs_data_path_1 = '../scidocs_data/paper_metadata_mag_mesh.json'
    scidocs_data_path_2 = '../scidocs_data/paper_metadata_recomm.json'
    scidocs_data_path_3 = '../scidocs_data/paper_metadata_view_cite_read.json'
    
    scidocs_vocab_save_path = 'data/scidocs_vocab.pkl'
    scidocs_doc_save_path = 'data/scidocs_doc.pkl'
    scidocs_seetopic_data_save_path = 'scidocs_seetopic.txt'

    # Seetopic Data
    seetopic_scidocs_data_path = '../seetopic_scidocs_data/scidocs.txt'

    scidocs_preproc = SeetopicScidocsPreprocess(seetopic_scidocs_data_path, id_prefix='SCIDOCS')
    scidocs_preproc.preprocess()
    scidocs_preproc.save_info(vocab_info_save_path=scidocs_vocab_save_path, doc_info_save_path=scidocs_doc_save_path, seetopic_data_save_path=scidocs_seetopic_data_save_path)

    #scidocs_preproc = SciDocsPreprocess(scidocs_data_path_1, scidocs_data_path_2, scidocs_data_path_3, id_prefix='SCIDOCS', sample=0.3)
    #scidocs_preproc.preprocess()
    #scidocs_preproc.save_info(save_path=scidocs_vocab_save_path)


    # preprocess amazon data
    amazon_data_path = '../amazon_data/amazon.txt'
    
    amazon_vocab_save_path = 'data/amazon_vocab.pkl'
    amazon_doc_save_path = 'data/amazon_doc.pkl'
    amazon_seetopic_data_save_path = 'amazon_seetopic.txt'

    amazon_preproc = AmazonPreprocess(amazon_data_path, id_prefix='AMAZON')
    amazon_preproc.preprocess()
    amazon_preproc.save_info(vocab_info_save_path=amazon_vocab_save_path, doc_info_save_path=amazon_doc_save_path, seetopic_data_save_path=amazon_seetopic_data_save_path)

    
    # preprocess french data
    french_news_data_path = '../news_fr_data/news_commentary_fr.txt'

    french_news_vocab_save_path = 'data/french_news_vocab.pkl'
    french_news_doc_save_path = 'data/french_news_doc.pkl'
    french_news_seetopic_data_save_path = 'french_news_seetopic.txt'

    french_news_preproc = FrenchNewsPreprocess(french_news_data_path, id_prefix='NEWS_FR')
    french_news_preproc.preprocess()
    french_news_preproc.save_info(vocab_info_save_path=french_news_vocab_save_path, doc_info_save_path=french_news_doc_save_path, seetopic_data_save_path=french_news_seetopic_data_save_path)


    # preprocess merged data
    merged_vocab_info_df_list = [scidocs_preproc.vocab_info_df, amazon_preproc.vocab_info_df, french_news_preproc.vocab_info_df]
    merged_sentences_list = scidocs_preproc.data + amazon_preproc.data + french_news_preproc.data

    merged_vocab_save_path = 'data/merged_vocab.pkl'
    merged_doc_save_path = 'data/merged_doc.pkl'
    merged_seetopic_data_save_path = 'merged_seetopic.txt'

    merged_preproc = MergedPreprocess(vocab_info_df_list=merged_vocab_info_df_list, sentences_lists=merged_sentences_list)
    merged_preproc.preprocess()
    merged_preproc.save_info(vocab_info_save_path=merged_vocab_save_path, doc_info_save_path=merged_doc_save_path, seetopic_data_save_path=merged_seetopic_data_save_path)
