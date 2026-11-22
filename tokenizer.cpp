#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <utility>
#include <random>

using namespace std;

bool loadtokenizer(std::map<std::pair<int, int>, int> &mergeRules, std::vector<std::pair<int, int>> &mergeidx)
{
    std::ifstream file("tokenizer.txt");

    if (!file.is_open())
    {
        std::cout << "Failed to load tokenizer\n";
        return false;
    }
    int a, b, id;
    char plus, equal;
    mergeidx.clear();
    mergeRules.clear();
    while (file >> a >> plus >> b >> equal >> id)
    {

        mergeidx.push_back({a, b});
        mergeRules[{a, b}] = id;
    }
    return true;
}


int bpe(vector<int> &tokens,
std::map<std::pair<int, int>, int>& mergeRules,
std::vector<std::pair<int, int>>& mergeidx,int vocab_limit)
{
    mergeRules.clear();
    mergeidx.clear();

    std::map<std::pair<int, int>, int> counts;

    int r = 0;
    
    while (true)
    {   
        int limit =256+r;
        if(limit> vocab_limit && vocab_limit!=0)break;
        if (tokens.size() < 2)
            break;

        int m_id = 256 + r;

        counts.clear();
        for (size_t i = 0; i + 1 < tokens.size(); i++)
        {
            int a = tokens[i];
            int b = tokens[i + 1];

            counts[std::make_pair(a, b)]++;
        }

        int maxcount = 0;
        std::pair<int, int> bestPair = {0, 0};

        for (const auto &entry : counts)
        {
            if (entry.second > maxcount)
            {
                maxcount = entry.second;
                bestPair = entry.first;
            }
        }
        if (maxcount < 2)
        {

            break;
        }
        mergeRules[bestPair] = m_id;
        mergeidx.push_back(bestPair);
        // merging
        std::vector<int> merged;

        for (size_t i = 0; i < tokens.size(); i++)
        {
            int a = tokens[i];

            bool doesnt_exists = i + 1 >= tokens.size();
            int b = (!doesnt_exists) ? tokens[i + 1] : 0;

            if (a == bestPair.first && b == bestPair.second && !doesnt_exists)
            {
                merged.push_back(m_id);
                i++;
            }
            else
            {
                merged.push_back(a);
            }
        }

        tokens = merged;
        r++;
        
    }
    std::cout<<"vocab size"<< 256+r<<'\n';
    return 256+mergeidx.size();
}

int encode(std::vector<int> &tokens,std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx)
{
   
    for (int m = 0; m < mergeidx.size(); m++)
    {   
        std::pair<int, int> m_pair = {mergeidx[m].first, mergeidx[m].second};

        std::vector<int> merged;
        for (size_t i = 0; i < tokens.size(); i++)
        {

            int a = tokens[i];

            bool out_of_bound = i + 1 >= tokens.size();
            int b = out_of_bound ? 0 : tokens[i + 1];
            std::pair<int, int> t_pair = {a, b};
            if (t_pair == m_pair && !out_of_bound)
            {
                int id = mergeRules.at({a, b});
                merged.push_back(id);
                i++;
               
            }
            else
            {
                merged.push_back(a);
            }
        }
        tokens = merged;
    }
    return 256+mergeidx.size();

}

void decode(std::vector<int> &tokens,std::map<std::pair<int, int>, int>& mergeRules,std::vector<std::pair<int, int>>& mergeidx)
{
    for (int m = static_cast<int>(mergeidx.size()) - 1; m >= 0; m--)
    {
        std::pair<int, int> m_pair = {mergeidx[m].first, mergeidx[m].second};

        int id = mergeRules[m_pair];

        std::vector<int> dmerge;

        for (int i = 0; i < tokens.size(); i++)
        {
            int a = tokens[i];

            if (a == id)
            {
                dmerge.push_back(mergeidx[m].first);
                dmerge.push_back(mergeidx[m].second);
            }
            else
            {
                dmerge.push_back(a);
            }
        }
        tokens = dmerge;
    }
}


void save_tokenizer(std::map<std::pair<int, int>, int>& mergeRules,
std::vector<std::pair<int, int>>& mergeidx){
    std::ofstream savefile("tokenizer.txt");
  
    for (const auto &pair : mergeidx)
    {
        savefile << pair.first << " + "
                 << pair.second << " = "
                 << mergeRules.at(pair) << '\n';
    }

    savefile.close();
}

void get_tokens(std::vector<int>& tokens){
tokens.clear();
 ifstream file("data.txt", std::ios::binary);

    if (!file.is_open())
    {
        printf("failed to open files \n");
        return ;
    }
    string text;
    string line;

    while (std::getline(file, line))
    {
        text += line;
        text += '\n';
    }

   
    for (size_t i = 0; i < text.size(); i++)
    {
        int id = static_cast<unsigned char>(text[i]);
        tokens.push_back(id);
    }
}


