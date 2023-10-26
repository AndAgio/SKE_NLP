

def words_from_tokens(tokenizer, tokens) -> list[str]:
    for key, _ in tokens.items():
        tokens[key][0] = tokens[key][0][1:-1]
    words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    indices_to_merge = [i for i, word in enumerate(words) if word.startswith("##")]
    if not len(indices_to_merge) == 0:
        for iterator in range(len(indices_to_merge)):
            index_to_merge = indices_to_merge[iterator]
            words[index_to_merge - 1] = words[index_to_merge - 1] + words[index_to_merge].strip('#')
            del words[index_to_merge]
            indices_to_merge = [i - 1 if i >= index_to_merge else i for i in indices_to_merge]
    return words


def reformat_impact_scores_dict(data: dict) -> dict:
    data = {item['label_name']: {item[list(item.keys())[1]][i]: item[list(item.keys())[2]][i]
                                 for i in range(len(item[list(item.keys())[1]]))}
            for _, item in data.items()}
    return data
