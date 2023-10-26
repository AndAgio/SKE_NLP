from skenlp.data.wrench import WrenchDataset


class YoutubeDataset(WrenchDataset):
    def __init__(self,
                 tokenizer,
                 folder='datasets',
                 split='train',
                 logger=None):
        super().__init__(drive_id='1EA7sBgai-ZVAE2dcK0UkR7K3JjCzz_zT', tokenizer=tokenizer, folder=folder,
                         name='youtube', split=split, logger=logger)
