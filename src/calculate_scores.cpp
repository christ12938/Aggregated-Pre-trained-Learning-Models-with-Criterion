#include <unordered_set>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>


using JSON = nlohmann::json;
using SCORE = std::vector<std::pair<std::string, double>>;
using RESULT = std::vector<
    std::tuple<
        std::string,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE,
        double, SCORE
    >
>;


/* For locking Results */
std::mutex mtx;

/* For Progress Checking */
std::atomic<int> progress(0);


auto comparePairs(const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
    return a.second > b.second; // for descending order
}


auto countTotalVocabs(const JSON &vocabEntries){
    std::cout << "Total Number of Vocabularies: " << vocabEntries.size() << std::endl;
    return static_cast<double>(vocabEntries.size());
}


auto countTotalDocs(const JSON &docEntries){
    std::cout << "Total Number of Documents: " << docEntries.size() << std::endl;
    return static_cast<double>(docEntries.size());
} 


auto countTotalWords(const JSON &vocabEntries){
    double wordCounts = 0.0;
    for (const auto &[key, value] : vocabEntries.items()) {
        wordCounts += static_cast<double>(value["count"]);
    }
    std::cout << "Total Number of Words: " << wordCounts << std::endl;
    return wordCounts;
}


auto getIntersectingKeys(const JSON &firstLevelId, const JSON &secondLevelId){
    const JSON *smallerJson = &firstLevelId;
    const JSON *largerJson = &secondLevelId;

    if (firstLevelId.size() > secondLevelId.size()){
        std::swap(smallerJson, largerJson);
    }

    std::unordered_set<std::string> intersectingKeys;

    for (const auto &[key, value] : smallerJson->items()){
        if (largerJson->contains(key)){
            intersectingKeys.insert(key);
        }
    }
    return intersectingKeys;
}


double sumScores(const SCORE &scores){
    double sum = 0.0;
    for (const auto &pair : scores) {
        sum += pair.second;
    }
    return sum;
}


auto calScores(const JSON &firstLevel, const JSON &secondLevel, const JSON &docEntries, 
               const double &totalVocabCount, const double &totalDocCount, 
               const double &totalWordCount){

    /* Get Intersecting Keys */
    const auto intersectingKeys = getIntersectingKeys(firstLevel["id"], secondLevel["id"]);

    /* Declare Scores */
    double pmi = 0.0, pmi_laplace, pmi_smoothing_laplace;
    double ppmi = 0.0, ppmi_delta = 0.0, ppmi_laplace;
    double npmi = -1.0, npmi_laplace;
    double wapmi_alpha_1_laplace, wapmi_alpha_1_smoothing_laplace;
    double wapmi_alpha_2_laplace, wapmi_alpha_2_smoothing_laplace;
    double wapmi_alpha_3_laplace, wapmi_alpha_3_smoothing_laplace;
    double wappmi_alpha_1 = 0.0, wappmi_alpha_1_delta = 0.0, wappmi_alpha_1_laplace;
    double wappmi_alpha_2 = 0.0, wappmi_alpha_2_delta = 0.0, wappmi_alpha_2_laplace;
    double wappmi_alpha_3 = 0.0, wappmi_alpha_3_delta = 0.0, wappmi_alpha_3_laplace;

    /* Laplace Smoothing */ 
    const double c_i_laplace = static_cast<double>(firstLevel["id"].size()) + 0.01;
    const double c_j_laplace = static_cast<double>(secondLevel["id"].size()) + 0.01;
    const double c_ij_laplace = static_cast<double>(intersectingKeys.size()) + 0.01;

    const double p_i_laplace = c_i_laplace / totalDocCount;
    const double p_j_laplace = c_j_laplace / totalDocCount;
    const double p_ij_laplace = c_ij_laplace / totalDocCount;
   
    /* Delta */
    double delta;

    if (intersectingKeys.size()) {
        /* Declare frequencies */ 
        const double c_i = static_cast<double>(firstLevel["id"].size());
        const double c_j = static_cast<double>(secondLevel["id"].size());
        const double c_ij = static_cast<double>(intersectingKeys.size());

        const double p_i = c_i / totalDocCount;
        const double p_j = c_j / totalDocCount;
        const double p_ij = c_ij / totalDocCount;

        /* Calculate Delta */
        const double minFreq = std::min(c_i, c_j); 
        delta = (c_ij / (c_ij + 1)) * (minFreq / (minFreq + 1));

        /* Calculate simple scores */
        pmi = log2(p_ij / (p_i * p_j)); 
        ppmi = std::max(pmi, 0.0);
        ppmi_delta = delta * ppmi;
        npmi = -(pmi / log2(p_ij));
        wappmi_alpha_1 = p_ij * ppmi;
        wappmi_alpha_1_delta = delta * wappmi_alpha_1;
    }


    /* Calculate laplace simple scores */ 
    pmi_laplace = log2(p_ij_laplace / (p_i_laplace * p_j_laplace)); 
    /* k = 3 */
    pmi_smoothing_laplace = log2((p_ij_laplace * p_ij_laplace * p_ij_laplace) / (p_i_laplace * p_j_laplace));
    ppmi_laplace = std::max(pmi_laplace, 0.0);
    npmi_laplace = -(pmi_laplace / log2(p_ij_laplace));
    wapmi_alpha_1_laplace = p_ij_laplace * pmi_laplace;
    wapmi_alpha_1_smoothing_laplace = p_ij_laplace * pmi_smoothing_laplace;
    wappmi_alpha_1_laplace = p_ij_laplace * ppmi_laplace;

    /* Calculate Alphas */
    const double alpha_2 = 1 / (totalWordCount - static_cast<double>(firstLevel["count"]));
    const double alpha_3 = 1 / (static_cast<double>(secondLevel["count"]) * totalVocabCount);

    /* Calculate P(wt | di) */
    double p_w_i_d_i = 0.0, p_w_i_d_i_laplace = 0.0;
    for (const auto &intersectKey : intersectingKeys) { 
        const double w_i_d_i_count = static_cast<double>(firstLevel["id"][intersectKey]);
        const double w_i_d_i_count_laplace = static_cast<double>(firstLevel["id"][intersectKey]) + 0.01;
        const double d_i_length = docEntries[intersectKey];
        p_w_i_d_i += (w_i_d_i_count / d_i_length);
        p_w_i_d_i_laplace += (w_i_d_i_count_laplace / d_i_length);
    }

    /* Calculate WANPMI Alpha 2 and Alpha 3 */
    if (intersectingKeys.size()) {
        wappmi_alpha_2 = alpha_2 * p_w_i_d_i * ppmi;
        wappmi_alpha_3 = alpha_3 * p_w_i_d_i * ppmi;
        wappmi_alpha_2_delta = alpha_2 * p_w_i_d_i * ppmi_delta;
        wappmi_alpha_3_delta = alpha_3 * p_w_i_d_i * ppmi_delta;
    }

    wapmi_alpha_2_laplace = alpha_2 * p_w_i_d_i_laplace * pmi_laplace;
    wapmi_alpha_2_smoothing_laplace = alpha_2 * p_w_i_d_i_laplace * pmi_smoothing_laplace;
    wappmi_alpha_2_laplace = alpha_2 * p_w_i_d_i_laplace * ppmi_laplace;
    wapmi_alpha_3_laplace = alpha_3 * p_w_i_d_i_laplace * pmi_laplace;
    wapmi_alpha_3_smoothing_laplace = alpha_3 * p_w_i_d_i_laplace * pmi_smoothing_laplace;
    wappmi_alpha_3_laplace = alpha_3 * p_w_i_d_i_laplace * ppmi_laplace;

    return std::make_tuple(pmi_laplace, pmi_smoothing_laplace,
                           ppmi, ppmi_delta, ppmi_laplace, 
                           npmi, npmi_laplace,
                           wapmi_alpha_1_laplace, wapmi_alpha_2_laplace, wapmi_alpha_3_laplace,
                           wapmi_alpha_1_smoothing_laplace, wapmi_alpha_2_smoothing_laplace, wapmi_alpha_3_smoothing_laplace,
                           wappmi_alpha_1, wappmi_alpha_2, wappmi_alpha_3,
                           wappmi_alpha_1_delta, wappmi_alpha_2_delta, wappmi_alpha_3_delta,
                           wappmi_alpha_1_laplace, wappmi_alpha_2_laplace, wappmi_alpha_3_laplace);
}


void processBatch(const int start, const int end, RESULT &results, const JSON &vocabEntries, 
                  const JSON &docEntries, const double totalVocabCount, const double totalDocCount, 
                  const double totalWordCount, const int topK) {

    /* Loop through W_i */
    auto startIt = vocabEntries.begin();
    std::advance(startIt, start);

    auto endIt = vocabEntries.begin();
    std::advance(endIt, end);

    for (auto it = startIt; it != endIt; it++) { 

        const auto &firstLevelVocab = it.key();
        const auto &firstLevelValue = it.value();
        SCORE pmi_laplace_scores,
              pmi_smoothing_laplace_scores,
              ppmi_scores,
              ppmi_delta_scores,
              ppmi_laplace_scores,
              npmi_scores,
              npmi_laplace_scores,
              wapmi_alpha_1_laplace_scores,
              wapmi_alpha_2_laplace_scores,
              wapmi_alpha_3_laplace_scores,
              wapmi_alpha_1_smoothing_laplace_scores,
              wapmi_alpha_2_smoothing_laplace_scores,
              wapmi_alpha_3_smoothing_laplace_scores,
              wappmi_alpha_1_scores,
              wappmi_alpha_2_scores,
              wappmi_alpha_3_scores,
              wappmi_alpha_1_delta_scores,
              wappmi_alpha_2_delta_scores,
              wappmi_alpha_3_delta_scores,
              wappmi_alpha_1_laplace_scores,
              wappmi_alpha_2_laplace_scores,
              wappmi_alpha_3_laplace_scores;

        /* Loop through W_j */
        for (const auto &[secondLevelVocab, secondLevelValue] : vocabEntries.items()){
            /* Skip same word */
            if (&firstLevelValue != &secondLevelValue){
                const auto &[pmi_laplace_score,
                             pmi_smoothing_laplace_score,
                             ppmi_score,
                             ppmi_delta_score,
                             ppmi_laplace_score,
                             npmi_score,
                             npmi_laplace_score,
                             wapmi_alpha_1_laplace_score,
                             wapmi_alpha_2_laplace_score,
                             wapmi_alpha_3_laplace_score,
                             wapmi_alpha_1_smoothing_laplace_score,
                             wapmi_alpha_2_smoothing_laplace_score,
                             wapmi_alpha_3_smoothing_laplace_score,
                             wappmi_alpha_1_score,
                             wappmi_alpha_2_score,
                             wappmi_alpha_3_score,
                             wappmi_alpha_1_delta_score,
                             wappmi_alpha_2_delta_score,
                             wappmi_alpha_3_delta_score,
                             wappmi_alpha_1_laplace_score,
                             wappmi_alpha_2_laplace_score,
                             wappmi_alpha_3_laplace_score] = calScores(firstLevelValue, secondLevelValue, docEntries, 
                                                                        totalVocabCount, totalDocCount, totalWordCount);
                       
                pmi_laplace_scores.emplace_back(secondLevelVocab, pmi_laplace_score);
                pmi_smoothing_laplace_scores.emplace_back(secondLevelVocab, pmi_smoothing_laplace_score);
                ppmi_scores.emplace_back(secondLevelVocab, ppmi_score);
                ppmi_delta_scores.emplace_back(secondLevelVocab, ppmi_delta_score);
                ppmi_laplace_scores.emplace_back(secondLevelVocab, ppmi_laplace_score);
                npmi_scores.emplace_back(secondLevelVocab, npmi_score);
                npmi_laplace_scores.emplace_back(secondLevelVocab, npmi_laplace_score);
                wapmi_alpha_1_laplace_scores.emplace_back(secondLevelVocab, wapmi_alpha_1_laplace_score);
                wapmi_alpha_2_laplace_scores.emplace_back(secondLevelVocab, wapmi_alpha_2_laplace_score);
                wapmi_alpha_3_laplace_scores.emplace_back(secondLevelVocab, wapmi_alpha_3_laplace_score);
                wapmi_alpha_1_smoothing_laplace_scores.emplace_back(secondLevelVocab, wapmi_alpha_1_smoothing_laplace_score);
                wapmi_alpha_2_smoothing_laplace_scores.emplace_back(secondLevelVocab, wapmi_alpha_2_smoothing_laplace_score);
                wapmi_alpha_3_smoothing_laplace_scores.emplace_back(secondLevelVocab, wapmi_alpha_3_smoothing_laplace_score);
                wappmi_alpha_1_scores.emplace_back(secondLevelVocab, wappmi_alpha_1_score);
                wappmi_alpha_2_scores.emplace_back(secondLevelVocab, wappmi_alpha_2_score);
                wappmi_alpha_3_scores.emplace_back(secondLevelVocab, wappmi_alpha_3_score);
                wappmi_alpha_1_delta_scores.emplace_back(secondLevelVocab, wappmi_alpha_1_delta_score);
                wappmi_alpha_2_delta_scores.emplace_back(secondLevelVocab, wappmi_alpha_2_delta_score);
                wappmi_alpha_3_delta_scores.emplace_back(secondLevelVocab, wappmi_alpha_3_delta_score);
                wappmi_alpha_1_laplace_scores.emplace_back(secondLevelVocab, wappmi_alpha_1_laplace_score);
                wappmi_alpha_2_laplace_scores.emplace_back(secondLevelVocab, wappmi_alpha_2_laplace_score);
                wappmi_alpha_3_laplace_scores.emplace_back(secondLevelVocab, wappmi_alpha_3_laplace_score);
            }
        }


        /* Sort the words */
        std::sort(pmi_laplace_scores.begin(), pmi_laplace_scores.end(), comparePairs);
        std::sort(pmi_smoothing_laplace_scores.begin(), pmi_smoothing_laplace_scores.end(), comparePairs);
        std::sort(ppmi_scores.begin(), ppmi_scores.end(), comparePairs);
        std::sort(ppmi_delta_scores.begin(), ppmi_delta_scores.end(), comparePairs);
        std::sort(ppmi_laplace_scores.begin(), ppmi_laplace_scores.end(), comparePairs);
        std::sort(npmi_scores.begin(), npmi_scores.end(), comparePairs);
        std::sort(npmi_laplace_scores.begin(), npmi_laplace_scores.end(), comparePairs);
        std::sort(wapmi_alpha_1_laplace_scores.begin(), wapmi_alpha_1_laplace_scores.end(), comparePairs);
        std::sort(wapmi_alpha_2_laplace_scores.begin(), wapmi_alpha_2_laplace_scores.end(), comparePairs);
        std::sort(wapmi_alpha_3_laplace_scores.begin(), wapmi_alpha_3_laplace_scores.end(), comparePairs);
        std::sort(wapmi_alpha_1_smoothing_laplace_scores.begin(), wapmi_alpha_1_smoothing_laplace_scores.end(), comparePairs);
        std::sort(wapmi_alpha_2_smoothing_laplace_scores.begin(), wapmi_alpha_2_smoothing_laplace_scores.end(), comparePairs);
        std::sort(wapmi_alpha_3_smoothing_laplace_scores.begin(), wapmi_alpha_3_smoothing_laplace_scores.end(), comparePairs);
        std::sort(wappmi_alpha_1_scores.begin(), wappmi_alpha_1_scores.end(), comparePairs);
        std::sort(wappmi_alpha_2_scores.begin(), wappmi_alpha_2_scores.end(), comparePairs);
        std::sort(wappmi_alpha_3_scores.begin(), wappmi_alpha_3_scores.end(), comparePairs);
        std::sort(wappmi_alpha_1_delta_scores.begin(), wappmi_alpha_1_delta_scores.end(), comparePairs);
        std::sort(wappmi_alpha_2_delta_scores.begin(), wappmi_alpha_2_delta_scores.end(), comparePairs);
        std::sort(wappmi_alpha_3_delta_scores.begin(), wappmi_alpha_3_delta_scores.end(), comparePairs);
        std::sort(wappmi_alpha_1_laplace_scores.begin(), wappmi_alpha_1_laplace_scores.end(), comparePairs);
        std::sort(wappmi_alpha_2_laplace_scores.begin(), wappmi_alpha_2_laplace_scores.end(), comparePairs);
        std::sort(wappmi_alpha_3_laplace_scores.begin(), wappmi_alpha_3_laplace_scores.end(), comparePairs);

        /* Get Top K Words */ 
        const SCORE pmi_laplace_firstK(pmi_laplace_scores.begin(), pmi_laplace_scores.begin() + topK);
        const SCORE pmi_smoothing_laplace_firstK(pmi_smoothing_laplace_scores.begin(), pmi_smoothing_laplace_scores.begin() + topK);
        const SCORE ppmi_firstK(ppmi_scores.begin(), ppmi_scores.begin() + topK);
        const SCORE ppmi_delta_firstK(ppmi_delta_scores.begin(), ppmi_delta_scores.begin() + topK);
        const SCORE ppmi_laplace_firstK(ppmi_laplace_scores.begin(), ppmi_laplace_scores.begin() + topK);
        const SCORE npmi_firstK(npmi_scores.begin(), npmi_scores.begin() + topK);
        const SCORE npmi_laplace_firstK(npmi_laplace_scores.begin(), npmi_laplace_scores.begin() + topK);
        const SCORE wapmi_alpha_1_laplace_firstK(wapmi_alpha_1_laplace_scores.begin(), wapmi_alpha_1_laplace_scores.begin() + topK);
        const SCORE wapmi_alpha_2_laplace_firstK(wapmi_alpha_2_laplace_scores.begin(), wapmi_alpha_2_laplace_scores.begin() + topK);
        const SCORE wapmi_alpha_3_laplace_firstK(wapmi_alpha_3_laplace_scores.begin(), wapmi_alpha_3_laplace_scores.begin() + topK);
        const SCORE wapmi_alpha_1_smoothing_laplace_firstK(wapmi_alpha_1_smoothing_laplace_scores.begin(), wapmi_alpha_1_smoothing_laplace_scores.begin() + topK);
        const SCORE wapmi_alpha_2_smoothing_laplace_firstK(wapmi_alpha_2_smoothing_laplace_scores.begin(), wapmi_alpha_2_smoothing_laplace_scores.begin() + topK);
        const SCORE wapmi_alpha_3_smoothing_laplace_firstK(wapmi_alpha_3_smoothing_laplace_scores.begin(), wapmi_alpha_3_smoothing_laplace_scores.begin() + topK);
        const SCORE wappmi_alpha_1_firstK(wappmi_alpha_1_scores.begin(), wappmi_alpha_1_scores.begin() + topK);
        const SCORE wappmi_alpha_2_firstK(wappmi_alpha_2_scores.begin(), wappmi_alpha_2_scores.begin() + topK);
        const SCORE wappmi_alpha_3_firstK(wappmi_alpha_3_scores.begin(), wappmi_alpha_3_scores.begin() + topK);
        const SCORE wappmi_alpha_1_delta_firstK(wappmi_alpha_1_delta_scores.begin(), wappmi_alpha_1_delta_scores.begin() + topK);
        const SCORE wappmi_alpha_2_delta_firstK(wappmi_alpha_2_delta_scores.begin(), wappmi_alpha_2_delta_scores.begin() + topK);
        const SCORE wappmi_alpha_3_delta_firstK(wappmi_alpha_3_delta_scores.begin(), wappmi_alpha_3_delta_scores.begin() + topK);
        const SCORE wappmi_alpha_1_laplace_firstK(wappmi_alpha_1_laplace_scores.begin(), wappmi_alpha_1_laplace_scores.begin() + topK);
        const SCORE wappmi_alpha_2_laplace_firstK(wappmi_alpha_2_laplace_scores.begin(), wappmi_alpha_2_laplace_scores.begin() + topK);
        const SCORE wappmi_alpha_3_laplace_firstK(wappmi_alpha_3_laplace_scores.begin(), wappmi_alpha_3_laplace_scores.begin() + topK);

        /* Calculate total Score */
        const auto total_pmi_laplace_score = sumScores(pmi_laplace_scores);
        const auto total_pmi_smoothing_laplace_score = sumScores(pmi_smoothing_laplace_scores);
        const auto total_ppmi_score = sumScores(ppmi_scores);
        const auto total_ppmi_delta_score = sumScores(ppmi_delta_scores);
        const auto total_ppmi_laplace_score = sumScores(ppmi_laplace_scores);
        const auto total_npmi_score = sumScores(npmi_scores);
        const auto total_npmi_laplace_score = sumScores(npmi_laplace_scores);
        const auto total_wapmi_alpha_1_laplace_score = sumScores(wapmi_alpha_1_laplace_scores);
        const auto total_wapmi_alpha_2_laplace_score = sumScores(wapmi_alpha_2_laplace_scores);
        const auto total_wapmi_alpha_3_laplace_score = sumScores(wapmi_alpha_3_laplace_scores);
        const auto total_wapmi_alpha_1_smoothing_laplace_score = sumScores(wapmi_alpha_1_smoothing_laplace_scores);
        const auto total_wapmi_alpha_2_smoothing_laplace_score = sumScores(wapmi_alpha_2_smoothing_laplace_scores);
        const auto total_wapmi_alpha_3_smoothing_laplace_score = sumScores(wapmi_alpha_3_smoothing_laplace_scores);
        const auto total_wappmi_alpha_1_score = sumScores(wappmi_alpha_1_scores);
        const auto total_wappmi_alpha_2_score = sumScores(wappmi_alpha_2_scores);
        const auto total_wappmi_alpha_3_score = sumScores(wappmi_alpha_3_scores);
        const auto total_wappmi_alpha_1_delta_score = sumScores(wappmi_alpha_1_delta_scores);
        const auto total_wappmi_alpha_2_delta_score = sumScores(wappmi_alpha_2_delta_scores);
        const auto total_wappmi_alpha_3_delta_score = sumScores(wappmi_alpha_3_delta_scores);
        const auto total_wappmi_alpha_1_laplace_score = sumScores(wappmi_alpha_1_laplace_scores);
        const auto total_wappmi_alpha_2_laplace_score = sumScores(wappmi_alpha_2_laplace_scores);
        const auto total_wappmi_alpha_3_laplace_score = sumScores(wappmi_alpha_3_laplace_scores);

        /* Append Results */
        {
            std::lock_guard<std::mutex> lock(mtx);
            
            results.emplace_back(firstLevelVocab,
                                 total_pmi_laplace_score, pmi_laplace_firstK,
                                 total_pmi_smoothing_laplace_score, pmi_smoothing_laplace_firstK,
                                 total_ppmi_score, ppmi_firstK,
                                 total_ppmi_delta_score, ppmi_delta_firstK,
                                 total_ppmi_laplace_score, ppmi_laplace_firstK,
                                 total_npmi_score, npmi_firstK,
                                 total_npmi_laplace_score, npmi_laplace_firstK,
                                 total_wapmi_alpha_1_laplace_score, wapmi_alpha_1_laplace_firstK,
                                 total_wapmi_alpha_2_laplace_score, wapmi_alpha_2_laplace_firstK,
                                 total_wapmi_alpha_3_laplace_score, wapmi_alpha_3_laplace_firstK,
                                 total_wapmi_alpha_1_smoothing_laplace_score, wapmi_alpha_1_smoothing_laplace_firstK,
                                 total_wapmi_alpha_2_smoothing_laplace_score, wapmi_alpha_2_smoothing_laplace_firstK,
                                 total_wapmi_alpha_3_smoothing_laplace_score, wapmi_alpha_3_smoothing_laplace_firstK,
                                 total_wappmi_alpha_1_score, wappmi_alpha_1_firstK,
                                 total_wappmi_alpha_2_score, wappmi_alpha_2_firstK,
                                 total_wappmi_alpha_3_score, wappmi_alpha_3_firstK,
                                 total_wappmi_alpha_1_delta_score, wappmi_alpha_1_delta_firstK,
                                 total_wappmi_alpha_2_delta_score, wappmi_alpha_2_delta_firstK,
                                 total_wappmi_alpha_3_delta_score, wappmi_alpha_3_delta_firstK,
                                 total_wappmi_alpha_1_laplace_score, wappmi_alpha_1_laplace_firstK,
                                 total_wappmi_alpha_2_laplace_score, wappmi_alpha_2_laplace_firstK,
                                 total_wappmi_alpha_3_laplace_score, wappmi_alpha_3_laplace_firstK);
        }
        progress++;
    }
}


void monitorProgress(const int totalProgress) {
    int lastProgress = progress.load(); 
    while (true) {
        if (progress.load() != lastProgress) {
            lastProgress = progress.load();
            std::cout << "\rProgress: " << lastProgress << "/" << totalProgress << std::flush;
            if (lastProgress == totalProgress) {
                std::cout << std::endl << "Finished Calculating Scores" << std::endl;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    }
}


auto processData(const JSON &vocabEntries, const JSON &docEntries, const int &topK, int numThreads){
    const auto totalVocabCount = countTotalVocabs(vocabEntries);
    const auto totalDocCount = countTotalDocs(docEntries);
    const auto totalWordCount = countTotalWords(vocabEntries);
    RESULT results;

    /* Threads information */
    if (totalVocabCount < numThreads) {
        numThreads = totalVocabCount;
    }
    const int batchSize = totalVocabCount / numThreads;
    std::vector<std::thread> threads;

    /* Append worker threads */
    for (int i = 0; i < numThreads; i++) {
        int start = i * batchSize;
        int end = (i == numThreads - 1) ? totalVocabCount : start + batchSize;
        threads.emplace_back(processBatch, start, end, std::ref(results), std::cref(vocabEntries), 
                             std::cref(docEntries), totalVocabCount, totalDocCount, totalWordCount, 
                             topK);
    }

    /* Append Monitor Thread */
    threads.emplace_back(monitorProgress, totalVocabCount);

    /* Wait for Batches to Finish */
    for (auto &thread : threads) {
        thread.join();
    }

    return results;    
}


auto createCandidateJson(const SCORE &content){
    JSON candidate;
    for (const auto &entry : content) {
        candidate[entry.first] = entry.second;
    }
    return candidate;
}


auto createJson(const RESULT &results) {
    std::vector<JSON> dataToWrite;
    std::cout << "Creating JSON ..." << std::endl;
    for (const auto &firstLevelTuple : results){
        JSON entry;
        entry["vocab"] = std::get<0>(firstLevelTuple);

        entry["total_pmi_laplace_score"] = std::get<1>(firstLevelTuple);
        entry["pmi_laplace_candidate"] = createCandidateJson(std::get<2>(firstLevelTuple));

        entry["total_pmi_smoothing_laplace_score"] = std::get<3>(firstLevelTuple);
        entry["pmi_smoothing_laplace_candidate"] = createCandidateJson(std::get<4>(firstLevelTuple));

        entry["total_ppmi_score"] = std::get<5>(firstLevelTuple);
        entry["ppmi_candidate"] = createCandidateJson(std::get<6>(firstLevelTuple));

        entry["total_ppmi_delta_score"] = std::get<7>(firstLevelTuple);
        entry["ppmi_delta_candidate"] = createCandidateJson(std::get<8>(firstLevelTuple));

        entry["total_ppmi_laplace_score"] = std::get<9>(firstLevelTuple);
        entry["ppmi_laplace_candidate"] = createCandidateJson(std::get<10>(firstLevelTuple));

        entry["total_npmi_score"] = std::get<11>(firstLevelTuple);
        entry["npmi_candidate"] = createCandidateJson(std::get<12>(firstLevelTuple));

        entry["total_npmi_laplace_score"] = std::get<13>(firstLevelTuple);
        entry["npmi_laplace_candidate"] = createCandidateJson(std::get<14>(firstLevelTuple));

        entry["total_wapmi_alpha_1_laplace_score"] = std::get<15>(firstLevelTuple);
        entry["wapmi_alpha_1_laplace_candidate"] = createCandidateJson(std::get<16>(firstLevelTuple));

        entry["total_wapmi_alpha_2_laplace_score"] = std::get<17>(firstLevelTuple);
        entry["wapmi_alpha_2_laplace_candidate"] = createCandidateJson(std::get<18>(firstLevelTuple));

        entry["total_wapmi_alpha_3_laplace_score"] = std::get<19>(firstLevelTuple);
        entry["wapmi_alpha_3_laplace_candidate"] = createCandidateJson(std::get<20>(firstLevelTuple));

        entry["total_wapmi_alpha_1_smoothing_laplace_score"] = std::get<21>(firstLevelTuple);
        entry["wapmi_alpha_1_smoothing_laplace_candidate"] = createCandidateJson(std::get<22>(firstLevelTuple));

        entry["total_wapmi_alpha_2_smoothing_laplace_score"] = std::get<23>(firstLevelTuple);
        entry["wapmi_alpha_2_smoothing_laplace_candidate"] = createCandidateJson(std::get<24>(firstLevelTuple));

        entry["total_wapmi_alpha_3_smoothing_laplace_score"] = std::get<25>(firstLevelTuple);
        entry["wapmi_alpha_3_smoothing_laplace_candidate"] = createCandidateJson(std::get<26>(firstLevelTuple));

        entry["total_wappmi_alpha_1_score"] = std::get<27>(firstLevelTuple);
        entry["wappmi_alpha_1_candidate"] = createCandidateJson(std::get<28>(firstLevelTuple));

        entry["total_wappmi_alpha_2_score"] = std::get<29>(firstLevelTuple);
        entry["wappmi_alpha_2_candidate"] = createCandidateJson(std::get<30>(firstLevelTuple));

        entry["total_wappmi_alpha_3_score"] = std::get<31>(firstLevelTuple);
        entry["wappmi_alpha_3_candidate"] = createCandidateJson(std::get<32>(firstLevelTuple));

        entry["total_wappmi_alpha_1_delta_score"] = std::get<33>(firstLevelTuple);
        entry["wappmi_alpha_1_delta_candidate"] = createCandidateJson(std::get<34>(firstLevelTuple));

        entry["total_wappmi_alpha_2_delta_score"] = std::get<35>(firstLevelTuple);
        entry["wappmi_alpha_2_delta_candidate"] = createCandidateJson(std::get<36>(firstLevelTuple));

        entry["total_wappmi_alpha_3_delta_score"] = std::get<37>(firstLevelTuple);
        entry["wappmi_alpha_3_delta_candidate"] = createCandidateJson(std::get<38>(firstLevelTuple));

        entry["total_wappmi_alpha_1_laplace_score"] = std::get<39>(firstLevelTuple);
        entry["wappmi_alpha_1_laplace_candidate"] = createCandidateJson(std::get<40>(firstLevelTuple));

        entry["total_wappmi_alpha_2_laplace_score"] = std::get<41>(firstLevelTuple);
        entry["wappmi_alpha_2_laplace_candidate"] = createCandidateJson(std::get<42>(firstLevelTuple));

        entry["total_wappmi_alpha_3_laplace_score"] = std::get<43>(firstLevelTuple);
        entry["wappmi_alpha_3_laplace_candidate"] = createCandidateJson(std::get<44>(firstLevelTuple));

        dataToWrite.emplace_back(std::move(entry));
    }
    return dataToWrite;
}


auto writeJsonFile(const std::string &outputFilePath, const std::vector<JSON> &dataToWrite){
    std::ofstream outputFile(outputFilePath);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return 1;
    }

    // Write the JSON object to the file
    std::cout << "Writing JSON File" << std::endl;
    for (const auto &data : dataToWrite) {
        outputFile << data.dump() << std::endl;
    }

    // Close the file
    outputFile.close();

    std::cout << "Completed." << std::endl;
    return 0;
}


auto parse_arguments(int argc, char* argv[]) {
    std::string vocabFilePath;
    std::string docFilePath;
    std::string outputFilePath;
    int topK, numThreads;

    for (int i = 1; i < argc; i += 2) {
        std::string_view arg(argv[i]);

        if (arg == "-v") {
            vocabFilePath = argv[i + 1];
        } else if (arg == "-d") {
            docFilePath = argv[i + 1];
        } else if (arg == "-o") {
            outputFilePath = argv[i + 1];
        } else if (arg == "-topK") {
            topK = std::stoi(argv[i + 1]);
        } else if (arg == "-T") {
            numThreads = std::stoi(argv[i + 1]);
        }
    }

    return std::make_tuple(vocabFilePath, docFilePath, outputFilePath, topK, numThreads);
}


int main(int argc, char* argv[]) {
    /* Get File Arguments  */
    const auto [vocabFilePath, docFilePath, outputFilePath, topK, numThreads] = parse_arguments(argc, argv);
    
    /* Open Files */
    std::ifstream vocabFile(vocabFilePath); 
    std::ifstream docFile(docFilePath);

    if (!vocabFile.is_open() || !docFile.is_open()) {
        std::cerr << "Failed to open the files." << std::endl;
        return 1;
    }

    JSON vocabEntries;
    JSON docEntries;

    std::cout << "Reading Vocab JSON ... " << std::endl;
    try {
        vocabFile >> vocabEntries;
        vocabFile.close();
    } catch (JSON::parse_error &e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        vocabFile.close();
        return 1;
    } catch (JSON::exception &e) {
        std::cerr << "JSON exception: " << e.what() << std::endl;
        vocabFile.close();
        return 1;
    }


    std::cout << "Reading Doc JSON ... " << std::endl;
    try {
        docFile >> docEntries;
        docFile.close();
    } catch (JSON::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        docFile.close();
        return 1;
    } catch (JSON::exception &e) {
        std::cerr << "JSON exception: " << e.what() << std::endl;
        docFile.close();
        return 1;
    }

    /* Process Data */
    const auto results = processData(vocabEntries, docEntries, topK, numThreads);
    const auto dataToWrite = createJson(results);
    return writeJsonFile(outputFilePath, dataToWrite);
}
