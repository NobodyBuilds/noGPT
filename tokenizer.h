#pragma once
#include <iostream>
#include <vector>
#include<map>
#include <utility>

 inline int bpe(std::vector<int> &tokens,std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx,int vocab_limit=0);
 inline void save_tokenizer(std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx);

inline int encode(std::vector<int> &tokens,std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx);

inline void decode(std::vector<int> &tokens,std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx);

inline bool loadtokenizer(std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx);



inline void get_tokens(std::vector<int>& tokens);