from skenlp.data.wrench import WrenchDataset


class AGNewsDataset(WrenchDataset):
    def __init__(self,
                 tokenizer,
                 folder='datasets',
                 split='train',
                 logger=None):
        super().__init__(drive_id='1Xrr4rCZIPvAcWhs4SZpCufTxJiIE1IDj', tokenizer=tokenizer, folder=folder,
                         name='agnews', split=split, logger=logger)
