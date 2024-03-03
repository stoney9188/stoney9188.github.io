---
title: Review_KeyBERT
mathjax: true
article_header:
  type: overlay
  image:
    src: /screenshot.jpg
---

# KeyBERT

원문: [https://www.preprints.org/manuscript/201908.0073/v1](https://www.preprints.org/manuscript/201908.0073/v1) 

### 요약

- Keyword(unigram, 한 단어) 또는 Keyphrase(N-gram, 여러 단어)를 추출하기 위한 supervised approach
- self-labelling 알고리즘을 함께 제안

### 내용

1. 필요성
    1. 키워드 추출은 긴 문단보다 짧은 문장에서 더 어렵다는 문제점 극복을 위해, 의미와 상황을(semantically and contextually) 고려한 방법 제안
    2. 지도 학습 방법이므로 labeled dataset이 필요한데, 이 데이터셋을 만드는 작업이 매우 비용이 많이 드는 작업이므로, 빠르게, 스스로 학습 데이터를 생성하는 알고리즘 제안
2. 제안 방법
    1. 전체 과정
        
        ![Untitled](KeyBERT%20649acf0bcade4c73a848c8d1e7472909/Untitled.png)
        
        - (분야에 상관없이)학습을 위한 데이터 수집
        - 특수 문자, 기호 등 데이터 정제
        - Self-Labelling
        - labelled corpus를 이용해 키워드인지 아닌지 분류하는 문제에 대한 Bidirectional LSTM 모델 학습
    2. Self-Labelling
        - 문장을 BERT 모델에 입력으로 넣어서 각 단어에 대해 벡터로 변환 (w_i)
        - 문장을 벡터로 표현하기 위해, 문장에 포함된 각 단어의 벡터들의 평균을 구해서 표현 (W)
        - 각 단어마다 문장 전체와 코사인 유사도 구하기
        
        $$
        sim_i = cos(w_i, W)
        $$
        
        - 문장 전체와 유사도가 높은 단어를 키워드 후보군으로 선택
        - bidirectional LSTM 모델의 학습을 위해 키워드인 단어는 1로 인코딩, 키워드가 아닌 단어는 0으로 인코딩하여 학습
3. 실험 방법 & 결과
    1. 비교 방법: RAKE [1], TextRank [2] 등
    2. 데이터셋: INSPEC dataset [3], DUC dataset [4]
    3. 짧은 문장에서 다른 방법과의 비교
        
        ![Untitled](KeyBERT%20649acf0bcade4c73a848c8d1e7472909/Untitled%201.png)
        
    
4. HuggingFace model
    1. 이 논문의 본문에서는 제안 방법을 KeyBERT라고 지칭하지는 않고 있음
    2. BERT 모델을 기반으로 단어와 문장의 임베딩 벡터를 추출하여 유사도를 비교한다는 간결한 개념을 사용하고, self-labelling 접근 방법으로 학습 데이터를 만들지 않아도 쉽게 이용할 수 있다는 장점이 있음
    3. 또한 다양한 분야의 사전 학습된 BERT 모델을 기반으로 키워드 추출이 가능하여 활용성이 높은 모델
    4. 예시: 의학 분야의 데이터를 기반으로 사전 학습된 BlueBERT 모델을 이용하여 KeyBERT를 적용
        
        ```python
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer, models
        from transformers import AutoModel, AutoTokenizer
        
        model_name = 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'
        
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        kw_model = KeyBERT(model=model)
        keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 3), stop_words=None, diversity=0.5)
        ```
        
        - 위의 예시에서, abstract 변수에 PubMed에서 찾은 의학 분야의 논문에서 아래와 같은 초록을 추출해 적용한 결과
            - 초록
            "Purpose\nZPR1 is a zinc finger-containing protein that plays a crucial role in neurodegenerative diseases, lipid metabolism disorders, and non-alcoholic fatty liver disease. However, the expression pattern, prognostic value, and treatment response of ZPR1 in pan-cancer and hepatocellular carcinoma (HCC) remain unclear.\n\n\nPatients and Methods\nPan-cancer expression profiles and relevant clinical data were acquired from UCSC Xena platform. Pan-cancer expression, epigenetic profile, and clinical correlation analysis for ZPR1 were performed. We next explored the prognostic significance and potential biological functions of ZPR1 in HCC. Furthermore, the relationship between ZPR1 and immune infiltration and treatment response was investigated. Finally, quantitative immunohistochemistry (IHC) analysis was applied to assess the correlation of ZPR1 expression and immune microenvironment in HCC tissues using Qupath software.\n\n\nResults\nZPR1 was differentially expressed in most tumor types and significantly up-regulated in HCC. ZPR1 showed hypo-methylated status in most tumors. Pan-cancer correlation analysis indicated that ZPR1 was closely associated with clinicopathological factors and TMB, MSI, and stemness index in HCC. High ZPR1 expression could be an independent risk factor for adverse prognosis in HCC. ZPR1 correlated with immune cell infiltration and therapeutic response. Finally, IHC results suggested that ZPR1 correlated with CD4, CD56, CD68, and PD-L1 expression and is a promising pathological diagnostic marker in HCC.\n\n\nConclusion\nImmune infiltrate-associated ZPR1 could be considered a novel negative prognostic biomarker for therapeutic response in HCC.”
            - 키워드 추출 결과
            [('quantitative immunohistochemistry ihc', 0.7811), ('in hcc zpr1', 0.7774), ('zpr1 was differentially', 0.7703), ('immunohistochemistry ihc analysis', 0.7667), ('hcc zpr1 correlated', 0.7664)]
        - 키워드 추출 결과가 조금 중복되는 부분이 있지만, 일반적인 BERT 모델이라면 포함하지 않을 수 있는 의학 용어를 고려하여 추출했다는 것을 관찰할 수 있음

### References

[1] [https://onlinelibrary.wiley.com/doi/10.1002/9780470689646.ch1](https://onlinelibrary.wiley.com/doi/10.1002/9780470689646.ch1)
[2] [https://aclanthology.org/W04-3252.pdf](https://aclanthology.org/W04-3252.pdf)

[3] [https://aclanthology.org/N18-2100/](https://aclanthology.org/N18-2100/)

[4] [https://cdn.aaai.org/AAAI/2008/AAAI08-136.pdf](https://cdn.aaai.org/AAAI/2008/AAAI08-136.pdf)
