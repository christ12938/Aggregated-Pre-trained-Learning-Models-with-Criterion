from scidocs_preproc import SciDocsPreprocess

#TODO: Change Paths
data_1_path = '../../scidocs/data/paper_metadata_mag_mesh.json'
data_2_path = '../../scidocs/data/paper_metadata_recomm.json'
data_3_path = '../../scidocs/data/paper_metadata_view_cite_read.json'

if __name__ == "__main__":

    # Initialize preproccess module #
    scidocs_preproc = SciDocsPreprocess(data_1_path, data_2_path, data_3_path)

    scidocs_preproc.prepare_dataset()
    scidocs_preproc.save_scidocs_data(save_path="../data/scidocs_data/scidocs_dataset_sentence_split_raw.pkl")
    scidocs_preproc.save_vocab_list(save_path="../data/scidocs_data/scidocs_vocab_no_punc_no_special_char_keep_apos_hyphens.pkl")
