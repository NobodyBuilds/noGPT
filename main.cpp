#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <utility>
#include <random>
#include "tokenizer.h"
#include "transformer.h"

std::map<std::pair<int, int>, int> mergeRules;
std::vector<std::pair<int, int>> mergeidx;
std::vector<int> tokens;
std::vector<float>output;

int main()
{
 int vocab_size=0;
    get_tokens(tokens);

    if(!loadtokenizer(mergeRules,mergeidx)){

         vocab_size =bpe(tokens, mergeRules, mergeidx);
    }else{
       vocab_size= encode(tokens,mergeRules,mergeidx);
    }

   transformermodel transformer;
   transformer.vocab_size=vocab_size;
   transformer.dim=8;
   transformer.numheads=2;
   transformer.max_seq_len=128;

   initilizemodel(transformer);

   forward(transformer,tokens,output);

   std::vector<int> a = {3, 5, 7};
std::vector<int> b = {3, 5, 9};

std::vector<float> outA, outB;

forward(transformer, a, outA);
forward(transformer, b, outB);
float maxDiff = 0.0f;

for (int d = 0; d < transformer.dim; d++) {
    int idx = 2 * transformer.dim + d;

    float diff = std::abs(outA[idx] - outB[idx]);

    if (diff > maxDiff)
        maxDiff = diff;
}

std::cout << "Difference: " << maxDiff << '\n';
}