#pragma once 
#include <iostream>
#include <vector>
#include <map>




 void embeddings(std::vector<int> &tokens, std::vector<float> &output, int dim, int vocab_size);

struct transformermodel{
    int vocab_size;
    int dim;
    int numheads;
    int max_seq_len;

    std::vector<float>tokenTable;
    std::vector<float> posTable;

    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;
    std::vector<float> Wo;
    std::vector<float> gamma;
    std::vector<float> beta;
};


 void initilizemodel(transformermodel& m);

 void forward(
    const transformermodel& model,
     std::vector<int>& tokens,
    std::vector<float>& finalOutput
);