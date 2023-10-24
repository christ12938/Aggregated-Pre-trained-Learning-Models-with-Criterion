#include <unordered_set>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>


using JSON = nlohmann::json;
using SCORE = std::vector<std::pair<std::string, double>>;
using RESULT = std::vector<std::tuple<std::string, double, SCORE, double, SCORE, double, SCORE, double, SCORE>>;


auto comparePairs(const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
    return a.second > b.second; // for descending order
}


auto countTotalVocabs(const JSON &vocabEntries){
    std::cout << "Total Number of Vocabularies" << vocabEntries.size() << std::endl;
    return static_cast<double>(vocabEntries.size());
}


auto countTotalDocs(const JSON &docEntries){
    std::cout << "Total Number of Documents" << docEntries.size() << std::endl;
    return static_cast<double>(docEntries.size());
} 


auto countTotalWords(const JSON &vocabEntries){
    double wordCounts = 0.0;
    for (const auto &[key, value] : vocabEntries.items()) {
        wordCounts += static_cast<double>(value["count"]);
    }
    std::cout << "Total Number of Words" << wordCounts << std::endl;
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

    /* TODO: Laplce coeficient */
    const auto c_i = static_cast<double>(firstLevel["id"].size());
    const auto c_i_laplace = static_cast<double>(firstLevel["id"].size()) + 0.01;
    const auto c_j = static_cast<double>(secondLevel["id"].size());
    const auto c_j_laplace = static_cast<double>(secondLevel["id"].size()) + 0.01; 

    /* Get Intersecting Keys */
    const auto intersectingKeys = getIntersectingKeys(firstLevel["id"], secondLevel["id"]);
    size_t c_ij = intersectingKeys.size();

    /* Declare Scores */
    double npmi = -1.0;
    double wanpmi_alpha_1, wanpmi_alpha_2, wanpmi_alpha_3 = 0.0;

    if (c_ij != 0){
        /* Calculate Simple NPMI and WANPMI */
        const double p_i = c_i / totalDocCount;
        const double p_j = c_j / totalDocCount;
        const double p_ij = static_cast<double>(c_ij) / totalDocCount;
        npmi = -(std::log(p_ij / (p_i * p_j)) / std::log(p_ij));
        wanpmi_alpha_1 = p_ij * npmi;

        /* Calculate Alphas */
        const double alpha_2 = 1 / (totalWordCount - static_cast<double>(secondLevel["count"]));
        const double alpha_3 = 1 / (static_cast<double>(secondLevel["count"]) * totalVocabCount);

        /* Calculate P(wt | di) */
        double p_w_i_d_i = 0.0;
        for (const auto &intersectKey : intersectingKeys) { 
            const double w_i_d_i_count = firstLevel["id"][intersectKey];
            const double d_i_length = docEntries[intersectKey];
            p_w_i_d_i += (w_i_d_i_count / d_i_length);
        }

        /* Calculate WANPMI Alpha 2 and Alpha 3 */
        wanpmi_alpha_2 = alpha_2 * p_w_i_d_i * npmi;
        wanpmi_alpha_3 = alpha_3 * p_w_i_d_i * npmi;
    }
    return std::make_tuple(npmi, wanpmi_alpha_1, wanpmi_alpha_2, wanpmi_alpha_3);
}


auto procData(const JSON &vocabEntries, const JSON &docEntries, const int &topK){
    const auto totalVocabCount = countTotalVocabs(vocabEntries);
    const auto totalDocCount = countTotalDocs(docEntries);
    const auto totalWordCount = countTotalWords(vocabEntries);
    RESULT results;
    auto progress = 1;

    /* Loop through W_i */ 
    for (const auto &[firstLevelVocab, firstLevelValue] : vocabEntries.items()){
        std::cout << "\rProgress: " << progress << "/" << vocabEntries.size() << std::flush;

        SCORE npmi_scores, wanpmi_alpha_1_scores, wanpmi_alpha_2_scores, wanpmi_alpha_3_scores;

        /* Loop through W_j */
        for (const auto &[secondLevelVocab, secondLevelValue] : vocabEntries.items()){
            /* Skip same word */
            if (&firstLevelValue != &secondLevelValue){
                const auto &[npmi_score, 
                             wanpmi_alpha_1_score, 
                             wanpmi_alpha_2_score, 
                             wanpmi_alpha_3_score] 
                                 = calScores(firstLevelValue, secondLevelValue, docEntries, 
                                             totalVocabCount, totalDocCount, totalWordCount);
                npmi_scores.emplace_back(secondLevelVocab, npmi_score);
                wanpmi_alpha_1_scores.emplace_back(secondLevelVocab, wanpmi_alpha_1_score);
                wanpmi_alpha_2_scores.emplace_back(secondLevelVocab, wanpmi_alpha_2_score);
                wanpmi_alpha_3_scores.emplace_back(secondLevelVocab, wanpmi_alpha_3_score);
            }
        }

        /* Sort the words */
        std::sort(npmi_scores.begin(), npmi_scores.end(), comparePairs);
        std::sort(wanpmi_alpha_1_scores.begin(), wanpmi_alpha_1_scores.end(), comparePairs);
        std::sort(wanpmi_alpha_2_scores.begin(), wanpmi_alpha_2_scores.end(), comparePairs);
        std::sort(wanpmi_alpha_3_scores.begin(), wanpmi_alpha_3_scores.end(), comparePairs);
       
        /* Get Top K Words */ 
        const SCORE npmi_firstK(npmi_scores.begin(), npmi_scores.begin() + topK);
        const SCORE wanpmi_alpha_1_firstK(wanpmi_alpha_1_scores.begin(), wanpmi_alpha_1_scores.begin() + topK);
        const SCORE wanpmi_alpha_2_firstK(wanpmi_alpha_2_scores.begin(), wanpmi_alpha_2_scores.begin() + topK);
        const SCORE wanpmi_alpha_3_firstK(wanpmi_alpha_3_scores.begin(), wanpmi_alpha_3_scores.begin() + topK);

        /* Calculate total Score */
        const auto total_npmi_score = sumScores(npmi_scores);
        const auto total_wanpmi_alpha_1_score = sumScores(wanpmi_alpha_1_scores);
        const auto total_wanpmi_alpha_2_score = sumScores(wanpmi_alpha_2_scores);
        const auto total_wanpmi_alpha_3_score = sumScores(wanpmi_alpha_3_scores);

        /* Append Results */
        results.emplace_back(firstLevelVocab, total_npmi_score, npmi_firstK, 
                             total_wanpmi_alpha_1_score, wanpmi_alpha_3_firstK, 
                             total_wanpmi_alpha_2_score, wanpmi_alpha_2_firstK, 
                             total_wanpmi_alpha_3_score, wanpmi_alpha_3_firstK);
        progress++;
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
        entry["total_npmi_score"] = std::get<1>(firstLevelTuple);
        entry["npmi_candidate"] = createCandidateJson(std::get<2>(firstLevelTuple));
        entry["total_wanpmi_alpha_1_score"] = std::get<3>(firstLevelTuple);
        entry["wanpmi_alpha_1_candidate"] = createCandidateJson(std::get<4>(firstLevelTuple));
        entry["total_wanpmi_alpha_2_score"] = std::get<5>(firstLevelTuple);
        entry["wanpmi_alpha_2_candidate"] = createCandidateJson(std::get<6>(firstLevelTuple));
        entry["total_wanpmi_alpha_3_score"] = std::get<7>(firstLevelTuple);
        entry["wanpmi_alpha_3_candidate"] = createCandidateJson(std::get<8>(firstLevelTuple));

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

    return 0;
}


auto parse_arguments(int argc, char* argv[]) {
    std::string vocabFilePath;
    std::string docFilePath;
    std::string outputFilePath;
    int topK;

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
        }
    }

    return std::make_tuple(vocabFilePath, docFilePath, outputFilePath, topK);
}


int main(int argc, char* argv[]) {
    /* Get File Arguments  */
    const auto [vocabFilePath, docFilePath, outputFilePath, topK] = parse_arguments(argc, argv);
    
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
    const auto results = procData(vocabEntries, docEntries, topK);
    const auto dataToWrite = createJson(results);
    return writeJsonFile(outputFilePath, dataToWrite);
}
