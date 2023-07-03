from scidocs_preproc import SciDocsPreprocess

#TODO: Change Paths
data_1_path = '../../scidocs/data/paper_metadata_mag_mesh.json'
data_2_path = '../../scidocs/data/paper_metadata_recomm.json'
data_3_path = '../../scidocs/data/paper_metadata_view_cite_read.json'
vocab_list = '../vocabs/vocab.txt'

if __name__ == "__main__":

    # Initialize preproccess module #
    scidocs_preproc_128 = SciDocsPreprocess(data_1_path, data_2_path, data_3_path, vocab_list, keep_no_nsp=True, max_len=128)
    scidocs_preproc_256 = SciDocsPreprocess(data_1_path, data_2_path, data_3_path, vocab_list, keep_no_nsp=True, max_len=256)
    scidocs_preproc_512 = SciDocsPreprocess(data_1_path, data_2_path, data_3_path, vocab_list, keep_no_nsp=True, max_len=512)

    # Save 128 #
    scidocs_preproc_128.prepare_dataset()
    scidocs_preproc_128.save_scidocs_data(save_path="../data/scidocs_data/scidocs_dataset_128_no_punc_no_nsp_keep_dot.csv")
    scidocs_preproc_128.save_vocab_list(save_path="../data/scidocs_data/scidocs_vocab_128_no_punc_no_nsp_keep_dot.csv")

    # Save 256 #
    scidocs_preproc_256.prepare_dataset()
    scidocs_preproc_256.save_scidocs_data(save_path="../data/scidocs_data/scidocs_dataset_256_no_punc_no_nsp_keep_dot.csv")
    scidocs_preproc_256.save_vocab_list(save_path="../data/scidocs_data/scidocs_vocab_256_no_punc_no_nsp_keep_dot.csv")

    # Save 512 #
    scidocs_preproc_512.prepare_dataset()
    scidocs_preproc_512.save_scidocs_data(save_path="../data/scidocs_data/scidocs_dataset_512_no_punc_no_nsp_keep_dot.csv")
    scidocs_preproc_512.save_vocab_list(save_path="../data/scidocs_data/scidocs_vocab_512_no_punc_no_nsp_keep_dot.csv")
