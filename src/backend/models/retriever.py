import tempfile

from .encoder import BertPath, BERTWrapper


class BiEncoderRetriever:
    def __init__(self, bert_type: BertPath) -> None:
        self.encoder = BERTWrapper(bert_type)
        self.embedding_folder = tempfile.gettempdir()
