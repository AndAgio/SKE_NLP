from skenlp.data.wrench import WrenchDataset


class TrecDataset(WrenchDataset):
    def __init__(self,
                 tokenizer,
                 folder='datasets',
                 split='train',
                 logger=None):
        super().__init__(drive_id='1CLSothu0KPSbGJw__U1C4bsfHodfm4BD', tokenizer=tokenizer, folder=folder, name='trec',
                         split=split, logger=logger)
