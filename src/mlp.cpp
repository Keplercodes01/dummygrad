#include "engine.h"
#include "ops.h"
#include "activations.h"
#include "loss.h"
#include "init.h"
#include "optimizers.h"
#include "scalar_ops.h"
#include "broadcasting.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include<map>
#include<set>
#include<fstream>

std::vector<std::string> load_words(const std::string& filename) {
    std::vector<std::string> words;
    std::ifstream file(filename);
    std::string line;
    while(std::getline(file, line)) {
        if(!line.empty()) words.push_back(line);
    }
    return words;
}
int main() {
    
    // hyperparameters
    int block_size = 3;
    int n_embd = 10;
    int n_hidden = 200;
    int vocab_size = 27;
    int batch_size = 32;
    
    std::cout << "dummygrad MLP" << std::endl;
    auto words = load_words("/content/dummygrad/src/names.txt");
    std::cout<< "loaded" << words.size() << "words" << std::endl;

    //build vocab
    std::map<char, int> stoi;
    std::map<int, char> itos;

    std::set<char> chars;
    for(auto& w : words) for(char c : w) chars.insert(c);

    int idx = 1;
    for(char c : chars) {
        stoi[c] = idx;
        itos[idx] = c;
        idx++;
    }
    stoi['.'] = 0;
    itos[0] = '.';
    int vocab_size = stoi.size();

    std::cout<"vocab size: " << vocab_size <<std::endl;
    
    return 0;
}
