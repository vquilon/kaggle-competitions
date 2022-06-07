import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel
import torch

from typing import List

from patent_phrase_similarity.data.analysis.visualize_datasets import plot_score_vs_test_score
from patent_phrase_similarity.data.transformation.cpc_datasets import Datasets, CPCDatasets


def get_similar_score_v1(model, context: str, similar_sentences: List[str]) -> List[float]:
    sentence_embeddings = model.encode([context] + similar_sentences)
    print("Sentences Shape", sentence_embeddings.shape)

    similarity_distance = cosine_similarity(
        [sentence_embeddings[0]],
        sentence_embeddings[1:]
    )[0]
    return similarity_distance

def get_similar_score_v2(tokenizer, model, context: str, similar_sentences: List[str]) -> List[float]:
    tokens = {'input_ids': [], 'attention_mask': []}

    _sentences = [context] + similar_sentences
    for sentence in sentences:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    outputs.keys()

    embeddings = outputs.last_hidden_state

    attention_mask = tokens['attention_mask']
    print("Attention Mask Tensor", attention_mask.shape)

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # print(mask.shape)

    masked_embeddings = embeddings * mask

    # Then we sum the remained of the embeddings along axis 1:
    summed = torch.sum(masked_embeddings, 1)

    # Then sum the number of values that must be given attention in each position of the tensor:
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)

    # Finally, we calculate the mean as the sum of the embedding activations summed
    # divided by the number of values that should be given attention in each position summed_mask:
    mean_pooled = summed / summed_mask

    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()

    # calculate
    similarity_distance = cosine_similarity(
        [mean_pooled[0]],
        mean_pooled[1:]
    )[0]

    return similarity_distance


if __name__ == '__main__':
    model_V1 = SentenceTransformer('bert-base-nli-mean-tokens')

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model_v2 = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    sentences = [
        "Three years later, the coffin was still full of Jello.",
        "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
        "The person box was packed with jelly many dozens of months later.",
        "He found a leprechaun in his walnut shell."
    ]
    # similar_v1 = get_similar_score_v1(sentences[0], sentences[1:])
    # similar_v2 = get_similar_score_v2(sentences[0], sentences[1:])
    #
    # print(similar_v1)
    # print(similar_v2)

    # context = [
    #     'HUMAN NECESSITIES',
    #     'AGRICULTURE',
    #     'FORESTRY',
    #     'ANIMAL HUSBANDRY',
    #     'HUNTING',
    #     'TRAPPING',
    #     'FISHING',
    #     'SOIL WORKING IN AGRICULTURE OR FORESTRY',
    #     'PARTS, DETAILS, OR ACCESSORIES OF AGRICULTURAL MACHINES OR IMPLEMENTS, IN GENERAL ',
    #     'PLANTING',
    #     'SOWING',
    #     'FERTILISING ',
    #     'HARVESTING',
    #     'MOWING',
    #     'PROCESSING OF HARVESTED PRODUCE',
    #     'HAY OR STRAW PRESSES',
    #     'DEVICES FOR STORING AGRICULTURAL OR HORTICULTURAL PRODUCE ',
    #     'HORTICULTURE',
    #     'CULTIVATION OF VEGETABLES, FLOWERS, RICE, FRUIT, VINES, HOPS OR SEAWEED',
    #     'FORESTRY',
    #     'WATERING ',
    #     'NEW PLANTS OR ',
    #     ' PROCESSES FOR OBTAINING THEM',
    #     'PLANT REPRODUCTION BY TISSUE CULTURE TECHNIQUES',
    #     'MANUFACTURE OF DAIRY PRODUCTS ',
    #     'ANIMAL HUSBANDRY',
    #     'CARE OF BIRDS, FISHES, INSECTS',
    #     'FISHING',
    #     'REARING OR BREEDING ANIMALS, NOT OTHERWISE PROVIDED FOR',
    #     'NEW BREEDS OF ANIMALS',
    #     'SHOEING OF ANIMALS',
    #     'CATCHING, TRAPPING OR SCARING OF ANIMALS ',
    #     'APPARATUS FOR THE DESTRUCTION OF NOXIOUS ANIMALS OR NOXIOUS PLANTS',
    #     'PRESERVATION OF BODIES OF HUMANS OR ANIMALS OR PLANTS OR PARTS THEREOF ',
    #     'BIOCIDES, e.g. AS DISINFECTANTS, AS PESTICIDES OR AS HERBICIDES ',
    #     'PEST REPELLANTS OR ATTRACTANTS',
    #     'PLANT GROWTH REGULATORS'
    # ]
    #
    # similar_vector = get_similar_score_v1('\n'.join(context), ['panel', 'panel frame'])
    # print(similar_vector)
    # print("Difference:", 100 * abs(similar_vector[0] - similar_vector[1]), "%")
    #
    # similar_vector = get_similar_score_v1('\n'.join(context), ['distributor pipe', 'pipe'])
    # print(similar_vector)
    # print("Difference:", 100 * abs(similar_vector[0] - similar_vector[1]), "%")
    #
    # similar_vector = get_similar_score_v1('\n'.join(context), ['distributor pipe', 'liquid channels'])
    # print(similar_vector)
    # print("Difference:", 100 * abs(similar_vector[0] - similar_vector[1]), "%")
    #
    # similar_vector = get_similar_score_v1('\n'.join(context), ['distributor pipe', 'optical pipe'])
    # print(similar_vector)
    # print("Difference:", 100 * abs(similar_vector[0] - similar_vector[1]), "%")

    datasets = Datasets()
    cpc_datasets = CPCDatasets()

    train_df = datasets.get_train_df()
    cpc_train_df = cpc_datasets.merge_with_df(train_df)


    cpc_train_df['test_score'] = cpc_train_df.apply(lambda row: get_similar_score_v1(model_V1, row['anchor'], [row['target']])[0], axis=1)
    # similar_vector = get_similar_score_v1('distributor pipe', ['optical pipe'])
    # print(similar_vector)
    # print("Difference:", 100 * abs(similar_vector[0] - similar_vector[1]), "%")

    plot_score_vs_test_score(cpc_train_df, column_a='score', column_b='test_score')
