#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <utility>
#include <random>
#include <cmath>
#include <limits>
#include "transformer.h"

void basic_embedding(std::vector<int> &tokens, const std::vector<float> &table, std::vector<float> &output, int dim)
{

    output.resize(tokens.size() * dim);

    for (int t = 0; t < tokens.size(); t++)
    {
        for (size_t d = 0; d < dim; d++)
        {
            output[t * dim + d] = table[tokens[t] * dim + d];
        }
    }
}

void positon_embeddings(
    std::vector<float> &output,
    const std::vector<float> &posTable,
    int dim)
{
    int seq_len = output.size() / dim;

    for (int t = 0; t < seq_len; t++)
    {
        for (int d = 0; d < dim; d++)
        {

            output[t * dim + d] += posTable[t * dim + d];
        }
    }
}

void linearprojection(
    const std::vector<float> &output,
    const std::vector<float> &Wq,
    std::vector<float> &Q,
    int dim)
{
    int seq_len = output.size() / dim;

    Q.resize(seq_len * dim);

    for (int t = 0; t < seq_len; t++)
    {
        for (int j = 0; j < dim; j++)
        {
            float sum = 0;

            for (int k = 0; k < dim; k++)
            {
                sum += output[t * dim + k] * Wq[k * dim + j];
            }

            Q[t * dim + j] = sum;
        }
    }
}

void attentionScores(
    const std::vector<float> &Q,
    const std::vector<float> &K,
    std::vector<float> &scores,
    int dim, int numheads, int head_dim)
{
    int seq_len = Q.size() / dim;

    scores.resize(numheads * seq_len * seq_len);
    for (int h = 0; h < numheads; h++)
    {
        for (int i = 0; i < seq_len; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                if (j > i)
                {
                    scores[h * seq_len * seq_len + i * seq_len + j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float sum = 0;

                for (int d = 0; d < head_dim; d++)
                {

                    sum += Q[i * dim + h * head_dim + d] * K[j * dim + h * head_dim + d];
                }

                scores[h * seq_len * seq_len + i * seq_len + j] = sum / std::sqrt(static_cast<float>(head_dim));
            }
        }
    }
}

void softmaxscores(std::vector<float> &scores, int seq_len, int numheads)
{

    for (int h = 0; h < numheads; h++)
    {

        for (int i = 0; i < seq_len; i++)
        {
            float maxScore = -std::numeric_limits<float>::infinity();

            for (int j = 0; j < seq_len; j++)
            {

                if (scores[h * seq_len * seq_len + i * seq_len + j] > maxScore)
                    maxScore = scores[h * seq_len * seq_len + i * seq_len + j];
            }

            float sum = 0.0f;

            for (int j = 0; j < seq_len; j++)
            {
                int idx = h * seq_len * seq_len + i * seq_len + j;

                scores[idx] = std::exp(scores[idx] - maxScore);
                sum += scores[idx];
            }
            for (size_t j = 0; j < seq_len; j++)
            {
                int idx = h * seq_len * seq_len + i * seq_len + j;

                scores[idx] /= sum;
            }
        }
    }
}

void attentionOutput(
    const std::vector<float> &scores,
    const std::vector<float> &V,
    std::vector<float> &result,
    int seq_len,
    int dim, int numheads, int head_dim)
{
    result.resize(seq_len * dim);

    for (int h = 0; h < numheads; h++)
    {

        for (int i = 0; i < seq_len; i++)
        {
            for (int d = 0; d < head_dim; d++)
            {

                float sum = 0.0f;

                for (int j = 0; j < seq_len; j++)
                {

                    sum += scores[h * seq_len * seq_len + i * seq_len + j] * V[j * dim + h * head_dim + d];
                }

                result[i * dim + h * head_dim + d] = sum;
            }
        }
    }
}

void layernorm(
    std::vector<float> &data,
    const std::vector<float> &gamma,
    const std::vector<float> &beta,
    int dim)
{
    int seq_len = data.size() / dim;

    for (int t = 0; t < seq_len; t++)
    {
        float mean = 0.0f;

        for (int d = 0; d < dim; d++)
        {
            mean += data[t * dim + d];
        }

        mean /= dim;

        float variance = 0.0f;

        for (int d = 0; d < dim; d++)
        {
            float diff = data[t * dim + d] - mean;

            variance+=diff*diff;
        }

        variance /= dim;

        float epsilon = 1e-5f;

for (int d = 0; d < dim; d++)
{
    int idx = t * dim + d;

   data[idx]= ((data[idx]-mean) /std::sqrt(variance+epsilon)) *gamma[d]+beta[d];  
  
}
    }
}

void initilizemodel(transformermodel &m)
{

    std::mt19937 rng(42);
    std::mt19937 rn3(43);
    std::mt19937 rn4(44);
    std::mt19937 rn6(46);
    std::mt19937 rn5(45);
    std::mt19937 rn7(47);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    m.tokenTable.resize(m.vocab_size * m.dim);
    m.posTable.resize(m.max_seq_len * m.dim);
    m.Wq.resize(m.dim * m.dim);
    m.Wv.resize(m.dim * m.dim);
    m.Wk.resize(m.dim * m.dim);
    m.Wo.resize(m.dim * m.dim);
    m.gamma.resize(m.dim,1.0f);
    m.beta.resize(m.dim,0.0f);

    for (size_t i = 0; i < m.tokenTable.size(); i++)
    {
        m.tokenTable[i] = dist(rng);
    }
    for (size_t i = 0; i < m.posTable.size(); i++)
    {
        m.posTable[i] = dist(rn3);
    }
    for (size_t i = 0; i < m.dim * m.dim; i++)
    {
        m.Wq[i] = dist(rn4);
        m.Wk[i] = dist(rn5);
        m.Wv[i] = dist(rn6);
        m.Wo[i] = dist(rn7);
    }
}

void forward(
    const transformermodel &m,
    std::vector<int> &tokens,
    std::vector<float> &finalOutput)
{
    std::vector<float> x;
    basic_embedding(tokens, m.tokenTable, x, m.dim);

    positon_embeddings(x, m.posTable, m.dim);

    std::vector<float> v;
    std::vector<float> k;
    std::vector<float> Q;

    linearprojection(x, m.Wq, Q, m.dim);
    linearprojection(x, m.Wk, k, m.dim);
    linearprojection(x, m.Wv, v, m.dim);

    int seq_len = tokens.size();
    int head_dim = m.dim / m.numheads;

    std::vector<float> scores;
    std::vector<float> results;

    attentionScores(Q, k, scores, m.dim, m.numheads, head_dim);

    softmaxscores(scores, seq_len, m.numheads);

    attentionOutput(scores, v, results, seq_len, m.dim, m.numheads, head_dim);

    linearprojection(results, m.Wo, finalOutput, m.dim);
    for (size_t i = 0; i < finalOutput.size(); i++)
    {

        finalOutput[i] += x[i];
    }

    layernorm(finalOutput,m.gamma,m.beta,m.dim);
}