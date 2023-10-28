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
#include <any>


using JSON = nlohmann::json;
using SCORE = std::vector<std::pair<std::string, double>>;
using RESULT_CONTENT = std::vector<std::any>; 
using RESULT = std::vector<RESULT_CONTENT>;


/* For locking Results */
std::mutex mtx;

/* For Progress Checking */
std::atomic<int> progress(0);

/* For Storing the keys */
const std::vector<std::string> dataNames = {
    "total_pmi_laplace_score",
    "pmi_laplace_candidate",
    "total_pmi_smoothing_laplace_score",
    "pmi_smoothing_laplace_candidate",
    "total_ppmi_score",
    "ppmi_candidate",
    "total_ppmi_delta_score",
    "ppmi_delta_candidate",
    "total_ppmi_laplace_score",
    "ppmi_laplace_candidate",
    "total_npmi_score",
    "npmi_candidate",
    "total_npmi_laplace_score",
    "npmi_laplace_candidate",
    "total_wapmi_alpha_1_laplace_score",
    "wapmi_alpha_1_laplace_candidate",
    "total_wapmi_alpha_2_laplace_score",
    "wapmi_alpha_2_laplace_candidate",
    "total_wapmi_alpha_3_laplace_score",
    "wapmi_alpha_3_laplace_candidate",
    "total_wapmi_alpha_1_smoothing_laplace_score",
    "wapmi_alpha_1_smoothing_laplace_candidate",
    "total_wapmi_alpha_2_smoothing_laplace_score",
    "wapmi_alpha_2_smoothing_laplace_candidate",
    "total_wapmi_alpha_3_smoothing_laplace_score",
    "wapmi_alpha_3_smoothing_laplace_candidate",
    "total_wappmi_alpha_1_score",
    "wappmi_alpha_1_candidate",
    "total_wappmi_alpha_2_score",
    "wappmi_alpha_2_candidate",
    "total_wappmi_alpha_3_score",
    "wappmi_alpha_3_candidate",
    "total_wappmi_alpha_1_delta_score",
    "wappmi_alpha_1_delta_candidate",
    "total_wappmi_alpha_2_delta_score",
    "wappmi_alpha_2_delta_candidate",
    "total_wappmi_alpha_3_delta_score",
    "wappmi_alpha_3_delta_candidate",
    "total_wappmi_alpha_1_laplace_score",
    "wappmi_alpha_1_laplace_candidate",
    "total_wappmi_alpha_2_laplace_score",
    "wappmi_alpha_2_laplace_candidate",
    "total_wappmi_alpha_3_laplace_score",
    "wappmi_alpha_3_laplace_candidate"};

const std::size_t dataSize = dataNames.size();



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

    return std::vector<double>{
        pmi_laplace, pmi_smoothing_laplace,
        ppmi, ppmi_delta, ppmi_laplace,
        npmi, npmi_laplace,
        wapmi_alpha_1_laplace, wapmi_alpha_2_laplace, wapmi_alpha_3_laplace,
        wapmi_alpha_1_smoothing_laplace, wapmi_alpha_2_smoothing_laplace, wapmi_alpha_3_smoothing_laplace,
        wappmi_alpha_1, wappmi_alpha_2, wappmi_alpha_3,
        wappmi_alpha_1_delta, wappmi_alpha_2_delta, wappmi_alpha_3_delta,
        wappmi_alpha_1_laplace, wappmi_alpha_2_laplace, wappmi_alpha_3_laplace
    };
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
        
        /* Create scores vector */
        std::vector<SCORE> scoresVector(dataSize);

        /* Loop through W_j */
        for (const auto &[secondLevelVocab, secondLevelValue] : vocabEntries.items()){
            /* Skip same word */
            if (&firstLevelValue != &secondLevelValue){
                const auto &calculatedScores = calScores(firstLevelValue, secondLevelValue, docEntries, 
                                                         totalVocabCount, totalDocCount, totalWordCount);
               
                /* Emplace scores to list */
                for (int i = 0; i < dataSize; i++) {
                    scoresVector[i].emplace_back(secondLevelVocab, calculatedScores[i]); 
                }
            }
        }

        /* Sort the words, calculate scores and get first K candidates */
        RESULT_CONTENT resultContent;

        resultContent.emplace_back(firstLevelVocab);
        for (int i = 0; i < dataSize; i++) {
            std::sort(scoresVector[i].begin(), scoresVector[i].end(), comparePairs);
            const SCORE firstKScores(scoresVector[i].begin(), 
                                     scoresVector[i].begin() + std::min(topK, static_cast<int>(scoresVector[i].size())));
            resultContent.emplace_back(sumScores(scoresVector[i]));
            resultContent.emplace_back(std::move(firstKScores));
        }

        /* Append Results */
        {
            std::lock_guard<std::mutex> lock(mtx);
            
            results.emplace_back(std::move(resultContent));
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
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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
    for (const auto &firstLevelVector : results){
        JSON entry;
        entry["vocab"] = std::any_cast<std::string>(firstLevelVector[0]);
        for (int i = 1; i < dataSize + 1; i += 2){
           entry[dataNames[i - 1]] = std::any_cast<double>(firstLevelVector[i]); 
           entry[dataNames[i]] = createCandidateJson(std::any_cast<SCORE>(firstLevelVector[i + 1]));
        }

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
