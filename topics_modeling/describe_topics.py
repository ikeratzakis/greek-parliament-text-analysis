from gensim.models import LdaMulticore
from gensim.corpora import MmCorpus
import pyLDAvis.gensim_models
import os

model_dir = "models"


def show_topics(model: LdaMulticore, corpus: MmCorpus, id2word: dict):
    topics_vis = pyLDAvis.gensim_models.prepare(model, corpus, id2word)
    pyLDAvis.save_html(topics_vis, "greek_parliament_topics.html")


if __name__ == "__main__":
    model = LdaMulticore.load(os.path.join(model_dir, "greek_parliament_model.model"))
    corpus = MmCorpus(os.path.join(model_dir, "greek_parliament_corpus.mm"))
    id2word = model.id2word

    show_topics(model, corpus, id2word)
