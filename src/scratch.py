# Combined Criteria
combined = CombineCriteria(vocab_info_path=vocab_path,
                                       doc_info_path=doc_info_path,
                                       categorized_words_paths=words_paths, 
                                       model_list=model_list,
                                       criteria=criteria, 
                                       top_k=top_k)
combined.combine_results()
combined.save_result(result_save_path=combined_result_save_path, 
                                 decision_save_path=combined_decision_save_path)
del combined

combined = CombineCriteria(vocab_info_path=vocab_path,
                                       doc_info_path=doc_info_path,
                                       categorized_words_paths=words_paths, 
                                       model_list=model_list,
                                       criteria=criteria, 
                                       top_k=top_k,
                                       tf_idf=True)
combined.combine_results()
combined.save_result(result_save_path=combined_idf_result_save_path, 
                                 decision_save_path=combined_idf_decision_save_path)
del combined


# Score Embeds

combined = CombineCriteria(vocab_info_path=vocab_path,
                                       doc_info_path=doc_info_path,
                                       categorized_words_paths=words_paths, 
                                       model_list=model_list,
                                       criteria=criteria, 
                                       top_k=top_k)
combined.combine_results()
combined.save_result(result_save_path=combined_result_save_path, 
                                 decision_save_path=combined_decision_save_path)
del combined

combined = CombineCriteria(vocab_info_path=vocab_path,
                                       doc_info_path=doc_info_path,
                                       categorized_words_paths=words_paths, 
                                       model_list=model_list,
                                       criteria=criteria, 
                                       top_k=top_k,
                                       tf_idf=True)
combined.combine_results()
combined.save_result(result_save_path=combined_idf_result_save_path, 
                                 decision_save_path=combined_idf_decision_save_path)
del combined

combined = CombineCriteria(vocab_info_path=vocab_path,
                                       doc_info_path=doc_info_path,
                                       categorized_words_paths=words_paths, 
                                       model_list=model_list,
                                       criteria=criteria, 
                                       top_k=top_k)
combined.combine_results()
combined.save_result(result_save_path=combined_result_save_path, 
                                 decision_save_path=combined_decision_save_path)
del combined

combined = CombineCriteria(vocab_info_path=vocab_path,
                                       doc_info_path=doc_info_path,
                                       categorized_words_paths=words_paths, 
                                       model_list=model_list,
                                       criteria=criteria, 
                                       top_k=top_k,
                                       tf_idf=True)
combined.combine_results()
combined.save_result(result_save_path=combined_idf_result_save_path, 
                                 decision_save_path=combined_idf_decision_save_path)
del combined

