#TODO: Fix packages naming and directories
import preprocessors.scidos_preproc
import preprocessors.amazon_preproc
import preprocessors.french_XLSum_2_0_preproc
import preprocessors.merged_preproc


if __name__ == '__main__':
    
    # preprocess scidocs data
    scidocs_data_path_1 = '../scidocs_data/paper_metadata_mag_mesh.json'
    scidocs_data_path_2 = '../scidocs_data/paper_metadata_recomm.json'
    scidocs_data_path_3 = '../scidocs_data/paper_metadata_view_cite_read.json'
    
    scidocs_vocab_save_path = 'data/scidocs_vocab.pkl'

    scidocs_preproc = SciDocsPreprocess(scidocs_data_1_path, scidocs_data_2_path, scidocs_data_3_path, id_prefix='SCIDOCS')
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

    xlsum_fr_preproc = French_XLSum_2_0_Preproccess(xlsum_fr_data_path_1, xlsum_fr_data_path_2, xlsum_data_path_3, id_prefix='XLSUM_FR')
    xlsum_fr_preproc.preprocess()
    xlsum_fr_preproc.save_vocab_info(save_path=xlsum_fr_vocab_save_path)


    # preprocess merged data
    merged_sentences = list(scidocs_preproc.data_dict.values()) + amazon_preproc.data + list(xlsum_fr_preproc.data_dict.values()) 

    merged_vocab_save_path = 'data/merged_vocab.pkl'

    merged_preproc = MergedPreprocess(sentences=merged_sentences, id_prefix='MERGED')
    merged_preproc.preprocess()
    merged_preproc.save_vocab_info(save_path=merged_vocab_save_path)
