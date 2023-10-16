import main_types as t


def get_name(s: t.Sentence):
    return s.person_name if s.person_id != -1 else t.unknown_person_name


def aggregate_sentences_no_embeddings(sentences: list[t.Sentence]):
    return " ".join([s.text for s in sentences if s.sentence_type == t.SentenceType.Sentence])


def aggregate_sentences_with_embeddings(sentences: list[t.Sentence]):
    return "\n".join([
        get_name(s) + ": " + s.text.replace("\n", "\n  ")
        for s in sentences if s.sentence_type == t.SentenceType.Sentence])
