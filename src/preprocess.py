#TODO: Fix packages naming and directories
from scidocs_preproc import SciDocsPreprocess
from amazon_preproc import AmazonPreprocess
from french_XLSum_2_0_preproc import French_XLSum_2_0_Preprocess
from merged_preproc import MergedPreprocess


if __name__ == '__main__':
    
    # preprocess scidocs data
    scidocs_data_path_1 = '../scidocs_data/paper_metadata_mag_mesh.json'
    scidocs_data_path_2 = '../scidocs_data/paper_metadata_recomm.json'
    scidocs_data_path_3 = '../scidocs_data/paper_metadata_view_cite_read.json'
    
    scidocs_vocab_save_path = 'data/scidocs_vocab.pkl'

    scidocs_preproc = SciDocsPreprocess(scidocs_data_path_1, scidocs_data_path_2, scidocs_data_path_3, id_prefix='SCIDOCS', sample=0.3)
    scidocs_preproc.preprocess()
    scidocs_preproc.save_vocab_info(save_path=scidocs_vocab_save_path)


    # preprocess amazon data
    amazon_data_path = '../amazon_data/amazon.txt'
    
    amazon_vocab_save_path = 'data/amazon_vocab.pkl'

    amazon_preproc = AmazonPreprocess(amazon_data_path, id_prefix='AMAZON')
    amazon_preproc.preprocess()
    amazon_preproc.save_vocab_info(save_path=amazon_vocab_save_path)

    
    # preprocess french data
    xlsum_fr_data_path_1 = '../xlsum_fr_data/french_train.jsonl'
    xlsum_fr_data_path_2 = '../xlsum_fr_data/french_test.jsonl'
    xlsum_fr_data_path_3 = '../xlsum_fr_data/french_val.jsonl'
    
    xlsum_fr_vocab_save_path = 'data/xlsum_fr_vocab.pkl'

    xlsum_fr_preproc = French_XLSum_2_0_Preprocess(xlsum_fr_data_path_1, xlsum_fr_data_path_2, xlsum_fr_data_path_3, id_prefix='XLSUM_FR')
    xlsum_fr_preproc.preprocess()
    xlsum_fr_preproc.save_vocab_info(save_path=xlsum_fr_vocab_save_path)


    # preprocess merged data
    merged_vocab_info_df_list = [scidocs_preproc.vocab_info_df, amazon_preproc.vocab_info_df, xlsum_fr_preproc.vocab_info_df]

    merged_vocab_save_path = 'data/merged_vocab.pkl'

    merged_preproc = MergedPreprocess(vocab_info_df_list=merged_vocab_info_df_list)
    merged_preproc.preprocess()
    merged_preproc.save_vocab_info(save_path=merged_vocab_save_path)
