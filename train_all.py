import subprocess


def train_all():
    datasets = ['sms', 'trec', 'youtube', 'moral_alm_false', 'moral_baltimore_false', 'moral_blm_false', 'moral_election_false', 'moral_metoo_false', 'moral_sandy_false', 'moral_together_false']
    for dataset in datasets:
        command = 'python train.py --train_dataset="{}" --test_dataset="{}" --model="bert" --epoch=3 --device=0 --out_folder="outs"'.format(dataset, dataset)
        print('Running command: {}'.format(command))
        subprocess.run([command], shell=True)


if __name__ == '__main__':
    train_all()
