from skenlp.data.wrench import WrenchDataset


class SmsDataset(WrenchDataset):
    def __init__(self,
                 tokenizer,
                 folder='datasets',
                 split='train',
                 logger=None):
        super().__init__(drive_id='1DFu6HWREb9Y8S1_npq6IiAjH8up94IRy', tokenizer=tokenizer, folder=folder, name='sms',
                         split=split, logger=logger)
